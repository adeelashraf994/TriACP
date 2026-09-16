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

from .stage1 import extract_stage1_embeddings, load_stage1_package
from .utils import binary_metrics, ensure_dir, get_device, save_json, seed_everything


class CancerTypeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(128, 64), dropout=0.2):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, int(h)), nn.ReLU(), nn.Dropout(dropout)]
            d = int(h)
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _fit_binary(Xtr, ytr, Xva, yva, cfg, device, seed):
    seed_everything(seed)
    model = CancerTypeMLP(Xtr.shape[1], cfg["hidden_dims"], float(cfg["dropout"])).to(device)
    ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=True)
    pos = max(int((ytr == 1).sum()), 1)
    neg = max(int((ytr == 0).sum()), 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    best_state, best_auc = None, -1.0
    Xv = torch.tensor(Xva, dtype=torch.float32, device=device)
    for _ in range(int(cfg["epochs"])):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=pos_weight)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xv)).cpu().numpy()
        auc = binary_metrics(yva, pv, float(cfg["threshold"]))["auc"]
        auc = -1.0 if np.isnan(auc) else auc
        if auc > best_auc:
            best_auc = auc
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict(model, X, device):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32, device=device))).cpu().numpy()


def train_stage2(config: dict):
    seed = int(config["seed"]); seed_everything(seed)
    device = get_device(config.get("device", "auto"))
    scfg = config["stage2"]; paths = config["paths"]
    stage1_model, tokenizer, aa_table, stage1_cfg = load_stage1_package(config["stage1"]["model_path"], device)

    df = pd.read_csv(paths["stage2"])
    if "sequence" not in df.columns:
        raise ValueError("Stage-2 CSV must contain a 'sequence' column.")
    cancer_types = list(scfg["cancer_types"])
    missing = [c for c in cancer_types if c not in df.columns]
    if missing:
        raise ValueError(f"Stage-2 wide CSV is missing cancer columns: {missing}")
    df = df.drop_duplicates(subset=["sequence"]).reset_index(drop=True)
    X = extract_stage1_embeddings(stage1_model, tokenizer, aa_table, df["sequence"].tolist(), stage1_cfg, device)

    out_dir = ensure_dir(Path(paths["output_dir"]) / "stage2")
    results, final_packages = {}, {}
    for cidx, cancer in enumerate(cancer_types):
        y = df[cancer].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            continue
        minority = int(np.bincount(y).min())
        n_splits = min(int(scfg["folds"]), minority)
        if n_splits < 2:
            continue
        rows = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(X, y), 1):
            fit, va = train_test_split(tr, test_size=0.2, stratify=y[tr], random_state=seed + fold)
            scaler = StandardScaler().fit(X[fit])
            Xfit, Xva, Xte = scaler.transform(X[fit]), scaler.transform(X[va]), scaler.transform(X[te])
            model = _fit_binary(Xfit, y[fit], Xva, y[va], scfg, device, seed + 100*cidx + fold)
            p = _predict(model, Xte, device)
            rows.append({"fold": fold, **binary_metrics(y[te], p, float(scfg["threshold"]))})
        results[cancer] = {
            metric: float(np.nanmean([r[metric] for r in rows]))
            for metric in ["accuracy", "precision", "recall", "f1", "mcc", "auc"]
        }

        fit, va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
        scaler = StandardScaler().fit(X[fit])
        model = _fit_binary(scaler.transform(X[fit]), y[fit], scaler.transform(X[va]), y[va], scfg, device, seed + 1000 + cidx)
        final_packages[cancer] = {
            "input_dim": X.shape[1], "hidden_dims": list(scfg["hidden_dims"]), "dropout": float(scfg["dropout"]),
            "scaler": scaler, "state_dict": {k: v.detach().cpu() for k,v in model.state_dict().items()},
            "threshold": float(scfg["threshold"]),
        }
        print("Stage-2", cancer, results[cancer])

    save_json(results, out_dir / "cv_summary.json")
    joblib.dump(final_packages, out_dir / "stage2_models.pkl")
    print("Saved Stage-2 outputs to", out_dir)
    return results
