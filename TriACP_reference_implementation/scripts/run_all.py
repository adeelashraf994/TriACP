from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from triacp.stage1 import train_stage1
from triacp.stage2 import train_stage2
from triacp.stage3 import train_stage3
from triacp.utils import load_config

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/baseline.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    train_stage1(cfg)
    train_stage2(cfg)
    train_stage3(cfg)
