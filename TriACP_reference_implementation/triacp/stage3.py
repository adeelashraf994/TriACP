from __future__ import annotations

import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .features import handcrafted_matrix
from .stage1 import extract_stage1_embeddings, load_stage1_package
from .utils import ensure_dir, get_device, multiclass_metrics, save_json, seed_everything


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class OrdinalPotencyClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, width=128, depth=2, dropout=0.2):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(), nn.LayerNorm(width), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(depth)])
        self.classifier = nn.Linear(width, num_classes)
        self.ordinal = nn.Linear(width, 1)

    def forward(self, x):
        h = self.blocks(self.input(x))
        return self.classifier(h), self.ordinal(h).squeeze(-1)


def _loss(logits, score, y, ordinal_weight):
    ce = F.cross_entropy(logits, y)
    probs = torch.softmax(logits, dim=1)
    values = torch.arange(logits.size(1), device=logits.device, dtype=torch.float32)
    expected = (probs * values.unsqueeze(0)).sum(dim=1)
    ordinal = F.mse_loss(expected, y.float()) + 0.25 * F.mse_loss(score, y.float())
    return ce + float(ordinal_weight) * ordinal


def _fit_one(Xtr, ytr, Xva, yva, cfg, num_classes, device, seed):
    seed_everything(seed)
    model = OrdinalPotencyClassifier(Xtr.shape[1], num_classes, int(cfg["width"]), int(cfg["depth"]), float(cfg["dropout"])).to(device)
    ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long))
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    best_state, best_f1 = None, -1.0
    Xv = torch.tensor(Xva, dtype=torch.float32, device=device)
    for _ in range(int(cfg["epochs"])):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); logits, score = model(xb)
            loss = _loss(logits, score, yb, cfg["ordinal_loss_weight"])
            loss.backward(); opt.step()
        probs, _ = _predict_members([model], Xva, device, mc_passes=1)
        f1 = multiclass_metrics(yva, probs)["macro_f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy({k: v.detach().cpu() for k,v in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict_members(models, X, device, mc_passes=1):
    x = torch.tensor(X, dtype=torch.float32, device=device)
    member_probs = []
    for model in models:
        passes = []
        for _ in range(max(1, int(mc_passes))):
            if mc_passes > 1:
                model.train()  # activates dropout for approximate epistemic uncertainty
            else:
                model.eval()
            with torch.no_grad():
                logits, _ = model(x)
                passes.append(torch.softmax(logits, dim=1).cpu().numpy())
        member_probs.append(np.mean(passes, axis=0))
    stack = np.stack(member_probs, axis=0)
    mean_prob = stack.mean(axis=0)
    uncertainty = stack.std(axis=0).mean(axis=1)
    return mean_prob, uncertainty


def _train_ensemble(Xtr, ytr, Xva, yva, cfg, num_classes, device, seed):
    return [
        _fit_one(Xtr, ytr, Xva, yva, cfg, num_classes, device, seed + i)
        for i in range(int(cfg["ensemble_size"]))
    ]


def train_stage3(config: dict):
    seed = int(config["seed"]); seed_everything(seed)
    device = get_device(config.get("device", "auto"))
    scfg = config["stage3"]; paths = config["paths"]
    stage1_model, tokenizer, aa_table, stage1_cfg = load_stage1_package(config["stage1"]["model_path"], device)

    df = pd.read_csv(paths["stage3"])
    if not {"sequence", "ic50"}.issubset(df.columns):
        raise ValueError("Stage-3 CSV must contain 'sequence' and 'ic50' columns.")
    df["ic50"] = pd.to_numeric(df["ic50"], errors="coerce")
    df = df[np.isfinite(df["ic50"]) & (df["ic50"] > 0)].drop_duplicates().reset_index(drop=True)
    thresholds = np.asarray(scfg["thresholds"], dtype=float)
    class_names = list(scfg["class_names"])
    if len(class_names) != len(thresholds) + 1:
        raise ValueError("class_names must contain len(thresholds)+1 entries.")
    y = np.digitize(df["ic50"].to_numpy(dtype=float), thresholds, right=False).astype(int)
    num_classes = len(class_names)

    X_transfer = extract_stage1_embeddings(stage1_model, tokenizer, aa_table, df["sequence"].tolist(), stage1_cfg, device)
    X_hc = handcrafted_matrix(df["sequence"].tolist(), aa_table)
    X = np.concatenate([X_transfer, X_hc], axis=1).astype(np.float32)

    counts = np.bincount(y, minlength=num_classes)
    n_splits = min(int(scfg["folds"]), int(counts[counts > 0].min()))
    if n_splits < 2:
        raise ValueError("Stage-3 needs at least two samples in each represented potency class.")
    rows = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        fit, va = train_test_split(tr, test_size=float(scfg["validation_fraction"]), stratify=y[tr], random_state=seed + fold)
        scaler = StandardScaler().fit(X[fit])
        models = _train_ensemble(scaler.transform(X[fit]), y[fit], scaler.transform(X[va]), y[va], scfg, num_classes, device, seed + 100*fold)
        probs, uncertainty = _predict_members(models, scaler.transform(X[te]), device, int(scfg["mc_dropout_passes"]))
        rows.append({"fold": fold, "uncertainty_mean": float(np.mean(uncertainty)), **multiclass_metrics(y[te], probs)})
        print("Stage-3 fold", fold, rows[-1])

    out_dir = ensure_dir(Path(paths["output_dir"]) / "stage3")
    save_json(rows, out_dir / "cv_metrics.json")
    save_json({"thresholds": thresholds.tolist(), "class_names": class_names}, out_dir / "class_config.json")

    fit, va = train_test_split(np.arange(len(y)), test_size=float(scfg["validation_fraction"]), stratify=y, random_state=seed)
    scaler = StandardScaler().fit(X[fit])
    models = _train_ensemble(scaler.transform(X[fit]), y[fit], scaler.transform(X[va]), y[va], scfg, num_classes, device, seed + 999)
    package = {
        "scaler": scaler,
        "class_names": class_names,
        "thresholds": thresholds.tolist(),
        "model_kwargs": {"input_dim": X.shape[1], "num_classes": num_classes, "width": int(scfg["width"]), "depth": int(scfg["depth"]), "dropout": float(scfg["dropout"])},
        "state_dicts": [{k: v.detach().cpu() for k,v in m.state_dict().items()} for m in models],
        "mc_dropout_passes": int(scfg["mc_dropout_passes"]),
    }
    joblib.dump(package, out_dir / "stage3_ensemble.pkl")
    print("Saved Stage-3 outputs to", out_dir)
    return rows
