from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from noise_cancellation.config import load_config
from noise_cancellation.data_prep.datasets import DATASET_NAMES, EXPERIMENT_ALIASES, normalize_dataset_name, prepare_all_data


def main() -> None:
    config = load_config(PROJECT_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("all", *DATASET_NAMES, *EXPERIMENT_ALIASES),
        default="all",
        help="dataset to prepare from local raw sources",
    )
    args = parser.parse_args()
    prepare_all_data(PROJECT_ROOT, config, dataset_name=normalize_dataset_name(args.dataset))


if __name__ == "__main__":
    main()
