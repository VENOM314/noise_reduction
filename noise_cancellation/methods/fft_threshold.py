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


def fft_denoise_windowed(
    noisy_signal: np.ndarray,
    sample_rate: int,
    *,
    keep_ratio: float,
    window_size: int,
    hop_size: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    if not (0 < keep_ratio <= 1):
        raise ValueError("keep_ratio must satisfy 0 < keep_ratio <= 1.")
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if hop_size <= 0:
        raise ValueError("hop_size must be positive.")

    audio = np.asarray(noisy_signal, dtype=np.float64)
    n_samples = len(audio)
    if n_samples == 0:
        raise ValueError("noisy_signal must not be empty.")

    scale = np.max(np.abs(audio)) + 1e-12
    work = audio / scale
    window_size = min(int(window_size), n_samples)
    hop_size = min(int(hop_size), window_size)
    window = np.hanning(window_size)
    if window_size == 1:
        window = np.ones(1)

    f_max = min(float(f_max), sample_rate / 2)
    freqs = np.fft.rfftfreq(window_size, d=1 / sample_rate)
    allowed = (freqs >= float(f_min)) & (freqs <= f_max)
    eligible_indices = np.flatnonzero(allowed)
    if eligible_indices.size == 0:
        eligible_indices = np.arange(len(freqs))
    k = max(1, int(round(keep_ratio * eligible_indices.size)))
    k = min(k, eligible_indices.size)

    output = np.zeros(n_samples)
    weight = np.zeros(n_samples)
    starts = list(range(0, max(n_samples - window_size + 1, 1), hop_size))
    if starts[-1] != n_samples - window_size:
        starts.append(n_samples - window_size)

    for start in starts:
        end = start + window_size
        spectrum = np.fft.rfft(work[start:end] * window)
        magnitudes = np.abs(spectrum[eligible_indices])
        kept_indices = eligible_indices[np.argsort(magnitudes)[-k:]]
        filtered_spectrum = np.zeros_like(spectrum)
        filtered_spectrum[kept_indices] = spectrum[kept_indices]
        reconstructed = np.fft.irfft(filtered_spectrum, n=window_size)
        output[start:end] += reconstructed * window
        weight[start:end] += window**2

    valid = weight > 1e-6
    output[valid] = output[valid] / weight[valid]
    output[~valid] = work[~valid]
    return output * scale


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
        window_size = int(self.params.get("window_size", len(audio)))
        denoised = fft_denoise_windowed(
            audio,
            sample_rate,
            keep_ratio=float(self.params["keep_ratio"]),
            window_size=window_size,
            hop_size=int(self.params.get("hop_size", window_size)),
            f_min=float(self.params.get("f_min", 0.0)),
            f_max=float(self.params.get("f_max", sample_rate / 2)),
        )
        peak = np.max(np.abs(denoised)) if denoised.size else 0.0
        if peak > 1.0:
            denoised = denoised / peak * 0.95
        return denoised.astype(np.float32)
