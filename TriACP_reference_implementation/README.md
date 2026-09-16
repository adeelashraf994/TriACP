# TriACP — Reference Implementation

This repository provides a **conceptual/reference implementation** of a three-stage computational framework for anticancer peptide (ACP) analysis:

1. **Stage 1 — ACP identification**  
   ESM-2 sequence representations are combined with BLOSUM62 and AAIndex-style residue descriptors, followed by capsule-based feature learning and an MLP classifier.
2. **Stage 2 — Cancer-type activity prediction**  
   Stage-1 representations are transferred to one-vs-all cancer-type classifiers.
3. **Stage 3 — IC50 potency classification**  
   Stage-1 representations are combined with biochemical descriptors and used by an ordinal, uncertainty-aware deep ensemble.

## Important reproducibility note

This public repository is intentionally a **reference implementation**, not a verbatim copy of the tuned experimental code used to obtain manuscript tables and figures. Paper-specific file paths, hardware settings, large-model checkpoints, extensive hyperparameter tuning, cancer-type-specific adjustments, and exact optimization settings are replaced by simple baseline defaults.

Therefore:

- the code reproduces the **methodological workflow and model concepts**;
- the supplied configuration is designed to be understandable and modifiable;
- the example CSV files are **schema/smoke-test examples only** and are not biological benchmark datasets;
- exact numerical results reported in a manuscript require the same datasets, preprocessing, splits, random seeds, model checkpoint, and tuned parameters used in the experiments.


## Repository layout

```text
TriACP_reference_implementation/
├── configs/
│   └── baseline.yaml
├── data/
│   ├── README.md
│   ├── aaindex_example.csv
│   ├── stage1_train_example.csv
│   ├── stage1_test_example.csv
│   ├── stage2_example.csv
│   └── stage3_example.csv
├── triacp/
│   ├── __init__.py
│   ├── utils.py
│   ├── features.py
│   ├── stage1.py
│   ├── stage2.py
│   └── stage3.py
├── scripts/
│   ├── train_stage1.py
│   ├── train_stage2.py
│   ├── train_stage3.py
│   └── run_all.py
├── requirements.txt
└── .gitignore
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\\Scripts\\activate      # Windows
pip install -r requirements.txt
pip install -e .
```

The baseline config uses the lightweight public checkpoint `facebook/esm2_t6_8M_UR50D` so the code can be inspected and tested more easily. You may replace it with another compatible ESM-2 checkpoint.

## Data format

See `data/README.md`. Replace the example paths in `configs/baseline.yaml` with your own datasets.

## Run each stage

```bash
python scripts/train_stage1.py --config configs/baseline.yaml
python scripts/train_stage2.py --config configs/baseline.yaml
python scripts/train_stage3.py --config configs/baseline.yaml
```

Or run the complete workflow:

```bash
python scripts/run_all.py --config configs/baseline.yaml
```

Stage 2 and Stage 3 load the saved Stage-1 model package specified by `stage1.model_path` in the YAML file.


## Outputs

By default, outputs are written under `outputs/`:

- Stage 1: CV metrics, independent-test metrics, and a saved ACP model package.
- Stage 2: per-cancer one-vs-all CV metrics and final baseline classifiers.
- Stage 3: potency CV metrics, class configuration, predictive uncertainty, and the final ensemble package.

## Reproducibility

The scripts seed Python, NumPy, and PyTorch. GPU kernels can still introduce small nondeterministic differences depending on hardware, CUDA, PyTorch, and transformer versions.

## Citation



## License


