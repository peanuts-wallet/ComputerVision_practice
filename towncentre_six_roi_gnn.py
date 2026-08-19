#!/usr/bin/env python3
"""Run the Town Centre ST-GNN with the corrected six-ROI configuration."""

from __future__ import annotations

import sys
from pathlib import Path

from towncentre_three_roi_gnn import main


PROJECT_DIR = Path(__file__).resolve().parent


if __name__ == "__main__":
    defaults = [
        "--config",
        str(PROJECT_DIR / "configs" / "towncentre_six_rois.json"),
        "--output-dir",
        str(PROJECT_DIR / "results" / "towncentre_six_roi"),
    ]
    sys.argv[1:1] = defaults
    raise SystemExit(main())
