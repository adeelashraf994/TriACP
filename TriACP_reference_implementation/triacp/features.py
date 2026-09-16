from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
BLOSUM62 = substitution_matrices.load("BLOSUM62")


def clean_sequence(sequence: str) -> str:
    s = str(sequence).upper().strip()
    return "".join(aa if aa in AA_ORDER else "X" for aa in s)


def load_aaindex_table(path: str | Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("AAIndex-style file needs an amino-acid column plus numeric feature columns.")
    aa_col = df.columns[0]
    value_cols = list(df.columns[1:])
    table = {}
    for _, row in df.iterrows():
        aa = str(row[aa_col]).strip().upper()
        vals = pd.to_numeric(row[value_cols], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        table[aa] = vals
    if not table:
        raise ValueError("AAIndex-style table is empty.")
    return table


def blosum_average(sequence: str) -> np.ndarray:
    s = clean_sequence(sequence)
    rows = []
    for aa in s:
        if aa not in AA_ORDER:
            rows.append(np.zeros(len(AA_ORDER), dtype=np.float32))
            continue
        rows.append(np.array([float(BLOSUM62[aa, b]) for b in AA_ORDER], dtype=np.float32))
    if not rows:
        return np.zeros(len(AA_ORDER), dtype=np.float32)
    return np.mean(np.stack(rows), axis=0).astype(np.float32)


def aaindex_average(sequence: str, table: dict[str, np.ndarray]) -> np.ndarray:
    dim = len(next(iter(table.values())))
    rows = [table.get(aa, np.zeros(dim, dtype=np.float32)) for aa in clean_sequence(sequence)]
    if not rows:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(np.stack(rows), axis=0).astype(np.float32)


def handcrafted_vector(sequence: str, table: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([blosum_average(sequence), aaindex_average(sequence, table)]).astype(np.float32)


def handcrafted_matrix(sequences, table: dict[str, np.ndarray]) -> np.ndarray:
    X = np.stack([handcrafted_vector(s, table) for s in sequences]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X
