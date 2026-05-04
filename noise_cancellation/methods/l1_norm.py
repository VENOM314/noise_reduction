from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import DenoisingMethod
from .least_squares import generate_frequency_grid


def build_sinusoidal_dictionary(
    sample_rate: int,
    window_size: int,
    frequencies: np.ndarray,
    normalize_columns: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(window_size) / sample_rate
    columns = []
    for frequency in frequencies:
        columns.append(np.cos(2 * np.pi * frequency * t))
        columns.append(np.sin(2 * np.pi * frequency * t))

    dictionary = np.column_stack(columns)
    column_norms = np.linalg.norm(dictionary, axis=0)
    column_norms[column_norms == 0] = 1.0
    if normalize_columns:
        dictionary = dictionary / column_norms
    else:
        column_norms = np.ones_like(column_norms)
    return dictionary, column_norms


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def l1_norm_denoise_fista(
    audio: np.ndarray,
    dictionary: np.ndarray,
    *,
    lambda_reg: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float64)
    lipschitz = np.linalg.norm(dictionary, ord=2) ** 2 + 1e-12
    step_size = 1.0 / lipschitz

    coefficients = np.zeros(dictionary.shape[1])
    momentum_coefficients = coefficients.copy()
    momentum = 1.0

    for _ in range(max_iter):
        old_coefficients = coefficients.copy()
        gradient = dictionary.T @ (dictionary @ momentum_coefficients - x)
        coefficients = soft_threshold(momentum_coefficients - step_size * gradient, lambda_reg * step_size)

        new_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
        momentum_coefficients = coefficients + ((momentum - 1.0) / new_momentum) * (coefficients - old_coefficients)
        momentum = new_momentum

        if np.linalg.norm(coefficients - old_coefficients) < tol:
            break

    return dictionary @ coefficients


def denoise_signal(
    audio: np.ndarray,
    sample_rate: int,
    *,
    grid_type: str,
    n_freqs: int,
    f_min: float,
    f_max: float,
    lambda_reg: float,
    max_iter: int,
    tol: float,
    window_size: int,
    hop_size: int,
    dictionary_cache: dict | None = None,
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

    key = (sample_rate, window_size, grid_type, n_freqs, float(f_min), float(f_max))
    if dictionary_cache is not None and key in dictionary_cache:
        dictionary = dictionary_cache[key]
    else:
        freqs = generate_frequency_grid(sample_rate, n_freqs, f_min, f_max, grid_type=grid_type)
        dictionary, _ = build_sinusoidal_dictionary(sample_rate, window_size, freqs, normalize_columns=True)
        if dictionary_cache is not None:
            dictionary_cache[key] = dictionary

    output = np.zeros(n_samples)
    weight = np.zeros(n_samples)
    starts = list(range(0, max(n_samples - window_size + 1, 1), hop_size))
    if starts[-1] != n_samples - window_size:
        starts.append(n_samples - window_size)

    for start in starts:
        end = start + window_size
        reconstructed = l1_norm_denoise_fista(
            work[start:end],
            dictionary,
            lambda_reg=lambda_reg,
            max_iter=max_iter,
            tol=tol,
        )
        output[start:end] += reconstructed * window
        weight[start:end] += window

    valid = weight > 1e-3
    output[valid] = output[valid] / weight[valid]
    output[~valid] = work[~valid]
    output = output * scale

    peak = np.max(np.abs(output)) if output.size else 0.0
    if peak > 1.0:
        output = output / peak * 0.95
    return output.astype(np.float32)


@dataclass
class L1NormMethod(DenoisingMethod):
    params: dict[str, Any]
    dictionary_cache: dict = field(default_factory=dict)

    name: str = "l1_norm"

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        dataset_name: str,
        overrides: dict[str, Any] | None = None,
    ) -> "L1NormMethod":
        presets = config["methods"][cls.name]["parameter_presets"]
        params = dict(presets[dataset_name])
        if overrides:
            params.update(overrides)
        return cls(params=params)

    def denoise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return denoise_signal(audio, sample_rate, dictionary_cache=self.dictionary_cache, **self.params)
