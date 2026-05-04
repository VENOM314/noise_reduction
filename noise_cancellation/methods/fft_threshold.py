from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import DenoisingMethod


def fft_denoise_top_k(noisy_signal: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(noisy_signal, dtype=np.float64)
    n_samples = len(x)
    if n_samples == 0:
        raise ValueError("noisy_signal must not be empty.")
    if k <= 0:
        raise ValueError("k must be positive.")

    spectrum = np.fft.rfft(x)
    k = min(k, len(spectrum))
    kept_indices = np.argsort(np.abs(spectrum))[-k:]

    filtered_spectrum = np.zeros_like(spectrum)
    filtered_spectrum[kept_indices] = spectrum[kept_indices]
    return np.fft.irfft(filtered_spectrum, n=n_samples), kept_indices


def fft_denoise_keep_ratio(noisy_signal: np.ndarray, keep_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    if not (0 < keep_ratio <= 1):
        raise ValueError("keep_ratio must satisfy 0 < keep_ratio <= 1.")
    n_unique = len(noisy_signal) // 2 + 1
    k = max(1, int(round(keep_ratio * n_unique)))
    return fft_denoise_top_k(noisy_signal, k)


@dataclass
class FFTThresholdMethod(DenoisingMethod):
    params: dict[str, Any]

    name: str = "fft_threshold"

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        dataset_name: str,
        overrides: dict[str, Any] | None = None,
    ) -> "FFTThresholdMethod":
        presets = config["methods"][cls.name]["parameter_presets"]
        params = dict(presets[dataset_name])
        if overrides:
            params.update(overrides)
        return cls(params=params)

    def denoise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        del sample_rate
        denoised, _ = fft_denoise_keep_ratio(audio, keep_ratio=float(self.params["keep_ratio"]))
        peak = np.max(np.abs(denoised)) if denoised.size else 0.0
        if peak > 1.0:
            denoised = denoised / peak * 0.95
        return denoised.astype(np.float32)
