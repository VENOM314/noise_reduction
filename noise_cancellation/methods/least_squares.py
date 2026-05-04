from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import DenoisingMethod


def generate_frequency_grid(
    sample_rate: int,
    n_freqs: int,
    f_min: float,
    f_max: float,
    grid_type: str = "linear",
) -> np.ndarray:
    f_max = min(f_max, sample_rate / 2)
    if grid_type == "linear":
        return np.linspace(f_min, f_max, n_freqs)
    if grid_type == "log":
        return np.logspace(np.log10(f_min), np.log10(f_max), n_freqs)
    raise ValueError(f"Unsupported grid_type: {grid_type}")


def build_design_matrix(sample_rate: int, window_size: int, freqs: np.ndarray) -> np.ndarray:
    t = np.arange(window_size) / sample_rate
    cols = []
    for freq in freqs:
        cols.append(np.cos(2 * np.pi * freq * t))
        cols.append(np.sin(2 * np.pi * freq * t))
    return np.column_stack(cols)


def top_k_reconstruct(coefficients: np.ndarray, design_matrix: np.ndarray, k: int) -> np.ndarray:
    amplitudes = np.sqrt(coefficients[0::2] ** 2 + coefficients[1::2] ** 2)
    k = min(k, len(amplitudes))
    top_indices = np.argsort(amplitudes)[-k:]
    sparse = np.zeros_like(coefficients)
    for index in top_indices:
        sparse[2 * index] = coefficients[2 * index]
        sparse[2 * index + 1] = coefficients[2 * index + 1]
    return design_matrix @ sparse


def denoise_signal(
    audio: np.ndarray,
    sample_rate: int,
    *,
    grid_type: str,
    n_freqs: int,
    f_min: float,
    f_max: float,
    K: int,
    window_size: int,
    hop_size: int,
    matrix_cache: dict | None = None,
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    scale = np.max(np.abs(audio)) + 1e-12
    work = audio / scale
    n_samples = len(work)

    window_size = min(window_size, n_samples)
    hop_size = min(hop_size, window_size)
    window = np.hanning(window_size)
    if window_size == 1:
        window = np.ones(1)

    freqs = generate_frequency_grid(sample_rate, n_freqs, f_min, f_max, grid_type=grid_type)
    key = (sample_rate, window_size, grid_type, n_freqs, float(f_min), float(f_max))
    if matrix_cache is not None and key in matrix_cache:
        design_matrix, design_pinv = matrix_cache[key]
    else:
        design_matrix = build_design_matrix(sample_rate, window_size, freqs)
        design_pinv = np.linalg.pinv(design_matrix)
        if matrix_cache is not None:
            matrix_cache[key] = (design_matrix, design_pinv)

    output = np.zeros(n_samples)
    weight = np.zeros(n_samples)
    starts = list(range(0, max(n_samples - window_size + 1, 1), hop_size))
    if starts[-1] != n_samples - window_size:
        starts.append(n_samples - window_size)

    for start in starts:
        end = start + window_size
        coefficients = design_pinv @ work[start:end]
        reconstructed = top_k_reconstruct(coefficients, design_matrix, K)
        output[start:end] += reconstructed * window
        weight[start:end] += window

    valid = weight > 1e-3
    output[valid] = output[valid] / weight[valid]
    output[~valid] = work[~valid]
    output = output * scale

    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak * 0.95
    return output.astype(np.float32)


@dataclass
class LeastSquaresMethod(DenoisingMethod):
    params: dict[str, Any]
    matrix_cache: dict = field(default_factory=dict)

    name: str = "least_squares"

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        dataset_name: str,
        overrides: dict[str, Any] | None = None,
    ) -> "LeastSquaresMethod":
        presets = config["methods"][cls.name]["parameter_presets"]
        params = dict(presets[dataset_name])
        if overrides:
            params.update(overrides)
        return cls(params=params)

    def denoise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return denoise_signal(audio, sample_rate, matrix_cache=self.matrix_cache, **self.params)
