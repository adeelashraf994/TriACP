from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from .features import handcrafted_matrix, load_aaindex_table
from .utils import binary_metrics, ensure_dir, get_device, save_json, seed_everything


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = list(map(str, sequences))
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        y = -1 if self.labels is None else int(self.labels[idx])
        return self.sequences[idx], y


class PrimaryCapsule(nn.Module):
    def __init__(self, in_dim: int, n_capsules: int = 8, capsule_dim: int = 8):
        super().__init__()
        self.n_capsules = n_capsules
        self.capsule_dim = capsule_dim
        self.linear = nn.Linear(in_dim, n_capsules * capsule_dim)

    def forward(self, x):
        return self.linear(x).view(x.size(0), self.n_capsules, self.capsule_dim)


class DigitCapsule(nn.Module):
    def __init__(self, in_caps: int, in_dim: int, out_caps: int = 2, out_dim: int = 8, routing: int = 2):
        super().__init__()
        self.routing = routing
        self.weight = nn.Parameter(torch.randn(1, in_caps, out_caps, in_dim, out_dim) * 0.02)

    @staticmethod
    def squash(x, dim=-1):
        norm = torch.norm(x, p=2, dim=dim, keepdim=True)
        scale = (norm ** 2 / (1.0 + norm ** 2)) / (norm + 1e-8)
        return scale * x

    def forward(self, u):
        bsz = u.size(0)
        u_hat = torch.matmul(u.unsqueeze(2).unsqueeze(3), self.weight.expand(bsz, -1, -1, -1, -1)).squeeze(3)
        logits = torch.zeros(bsz, u_hat.size(1), u_hat.size(2), 1, device=u.device)
        for i in range(self.routing):
            coeff = torch.softmax(logits, dim=2)
            v = self.squash((coeff * u_hat).sum(dim=1))
            if i < self.routing - 1:
                logits = logits + (u_hat * v.unsqueeze(1)).sum(dim=-1, keepdim=True)
        return v


class HybridACPModel(nn.Module):
    """Baseline ESM-2 + BLOSUM62/AAIndex + capsule + MLP model."""

    def __init__(self, cfg: dict, handcrafted_dim: int):
        super().__init__()
        self.cfg = cfg
        self.esm = AutoModel.from_pretrained(cfg["model_name"])
        if cfg.get("freeze_esm", True):
            for p in self.esm.parameters():
                p.requires_grad = False

        esm_dim = int(self.esm.config.hidden_size)
        feat_hidden = int(cfg.get("feature_hidden", 64))
        dropout = float(cfg.get("dropout", 0.2))
        self.feature_proj = nn.Sequential(
            nn.Linear(handcrafted_dim, feat_hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        combined_dim = esm_dim + feat_hidden
        n_primary = int(cfg.get("primary_capsules", 8))
        primary_dim = int(cfg.get("primary_dim", 8))
        digit_dim = int(cfg.get("digit_dim", 8))
        self.primary = PrimaryCapsule(combined_dim, n_primary, primary_dim)
        self.digit = DigitCapsule(
            n_primary, primary_dim, out_caps=2, out_dim=digit_dim,
            routing=int(cfg.get("routing_iterations", 2)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * digit_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 2)
        )

    @staticmethod
    def _masked_mean(hidden, mask):
        m = mask.float().unsqueeze(-1)
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def encode(self, input_ids, attention_mask, handcrafted):
        out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        seq = self._masked_mean(out.last_hidden_state, attention_mask)
        hc = self.feature_proj(handcrafted)
        combined = torch.cat([seq, hc], dim=1)
        digit = self.digit(self.primary(combined)).flatten(1)
        transferred = torch.cat([combined, digit], dim=1)
        return digit, transferred

    def forward(self, input_ids, attention_mask, handcrafted):
        digit, _ = self.encode(input_ids, attention_mask, handcrafted)
        return self.classifier(digit)


def _collate(batch, tokenizer, aa_table, max_length, device):
    seqs = [x[0] for x in batch]
    labels = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)
    tok = tokenizer(seqs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    tok = {k: v.to(device) for k, v in tok.items()}
    hc = torch.tensor(handcrafted_matrix(seqs, aa_table), dtype=torch.float32, device=device)
    return seqs, labels, tok, hc


def _predict(model, loader, tokenizer, aa_table, cfg, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            _, y, tok, hc = _collate(batch, tokenizer, aa_table, int(cfg["max_length"]), device)
            p = torch.softmax(model(tok["input_ids"], tok["attention_mask"], hc), dim=1)[:, 1]
            ys.extend(y.cpu().numpy().tolist())
            ps.extend(p.cpu().numpy().tolist())
    return np.asarray(ys), np.asarray(ps)


def _fit(model, train_ds, val_ds, tokenizer, aa_table, cfg, device):
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]) * 2, shuffle=False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    best_state, best_mcc = None, -1e9
    for _ in range(int(cfg["epochs"])):
        model.train()
        for batch in train_loader:
            _, y, tok, hc = _collate(batch, tokenizer, aa_table, int(cfg["max_length"]), device)
            opt.zero_grad()
            logits = model(tok["input_ids"], tok["attention_mask"], hc)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
        yv, pv = _predict(model, val_loader, tokenizer, aa_table, cfg, device)
        score = binary_metrics(yv, pv)["mcc"]
        if score > best_mcc:
            best_mcc = score
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def load_stage1_package(model_path: str | Path, device=None):
    try:
        package = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        package = torch.load(model_path, map_location="cpu")
    cfg = package["stage1_config"]
    aa_table = {aa: np.asarray(vals, dtype=np.float32) for aa, vals in package["aaindex_table"].items()}
    handcrafted_dim = 20 + len(next(iter(aa_table.values())))
    model = HybridACPModel(cfg, handcrafted_dim)
    model.load_state_dict(package["state_dict"])
    device = device or get_device("auto")
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    return model, tokenizer, aa_table, cfg


def extract_stage1_embeddings(model, tokenizer, aa_table, sequences, cfg, device=None, batch_size=32):
    device = device or next(model.parameters()).device
    ds = SequenceDataset(sequences, np.zeros(len(sequences), dtype=int))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    feats = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            seqs, _, tok, hc = _collate(batch, tokenizer, aa_table, int(cfg["max_length"]), device)
            _, rep = model.encode(tok["input_ids"], tok["attention_mask"], hc)
            feats.append(rep.cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def train_stage1(config: dict):
    seed = int(config["seed"])
    seed_everything(seed)
    device = get_device(config.get("device", "auto"))
    scfg = config["stage1"]
    paths = config["paths"]
    aa_table = load_aaindex_table(paths["aaindex"])
    handcrafted_dim = 20 + len(next(iter(aa_table.values())))
    tokenizer = AutoTokenizer.from_pretrained(scfg["model_name"])

    train_df = pd.read_csv(paths["stage1_train"]).drop_duplicates(subset=["sequence", "label"])
    test_df = pd.read_csv(paths["stage1_test"]).drop_duplicates(subset=["sequence", "label"])
    train_df["sequence"] = train_df["sequence"].astype(str).str.upper()
    test_df["sequence"] = test_df["sequence"].astype(str).str.upper()
    overlap = set(train_df["sequence"]) & set(test_df["sequence"])
    if overlap:
        test_df = test_df[~test_df["sequence"].isin(overlap)].reset_index(drop=True)

    y = train_df["label"].astype(int).to_numpy()
    n_splits = min(int(scfg["folds"]), int(np.bincount(y).min()))
    if n_splits < 2:
        raise ValueError("Stage-1 needs at least two samples in each class for CV.")
    cv_rows = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(train_df, y), 1):
        seed_everything(seed + fold)
        model = HybridACPModel(scfg, handcrafted_dim).to(device)
        tr_ds = SequenceDataset(train_df.iloc[tr]["sequence"], y[tr])
        va_ds = SequenceDataset(train_df.iloc[va]["sequence"], y[va])
        model = _fit(model, tr_ds, va_ds, tokenizer, aa_table, scfg, device)
        va_loader = DataLoader(va_ds, batch_size=int(scfg["batch_size"]) * 2, shuffle=False)
        yt, pt = _predict(model, va_loader, tokenizer, aa_table, scfg, device)
        row = {"fold": fold, **binary_metrics(yt, pt)}
        cv_rows.append(row)
        print("Stage-1 fold", fold, row)

    fit_df, val_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df["label"], random_state=seed
    )
    model = HybridACPModel(scfg, handcrafted_dim).to(device)
    model = _fit(
        model,
        SequenceDataset(fit_df["sequence"], fit_df["label"]),
        SequenceDataset(val_df["sequence"], val_df["label"]),
        tokenizer, aa_table, scfg, device,
    )
    test_ds = SequenceDataset(test_df["sequence"], test_df["label"])
    test_loader = DataLoader(test_ds, batch_size=int(scfg["batch_size"]) * 2, shuffle=False)
    yt, pt = _predict(model, test_loader, tokenizer, aa_table, scfg, device)
    test_metrics = binary_metrics(yt, pt)

    out = ensure_dir(Path(paths["output_dir"]) / "stage1")
    save_json(cv_rows, out / "cv_metrics.json")
    save_json(test_metrics, out / "independent_test_metrics.json")
    package = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "stage1_config": scfg,
        "aaindex_table": {aa: vals.astype(float).tolist() for aa, vals in aa_table.items()},
    }
    model_path = Path(scfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, model_path)
    print("Stage-1 independent test:", test_metrics)
    print("Saved Stage-1 model:", model_path)
    return model_path
