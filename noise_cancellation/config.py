from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(root: Path | None = None) -> dict[str, Any]:
    root = (root or Path.cwd()).resolve()
    config_path = root / "configs" / "experiment_parameters.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def output_root(root: Path | None = None) -> Path:
    root = (root or Path.cwd()).resolve()
    return root / "outputs" / "experiments"
