from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class DenoisingMethod(ABC):
    """Common interface for denoising methods used by experiment runners."""

    name: str

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: dict[str, Any],
        dataset_name: str,
        overrides: dict[str, Any] | None = None,
    ) -> "DenoisingMethod":
        """Create a method instance from project configuration."""

    @abstractmethod
    def denoise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return a denoised mono audio signal."""
