from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import DenoisingMethod
from .least_squares import generate_frequency_grid
from .l1_norm import build_sinusoidal_dictionary


def omp_reconstruct(
    frame: np.ndarray,
    dictionary: np.ndarray,
    *,
    k: int,
    tol: float,
) -> np.ndarray:
    """
    Pair-OMP on a sine/cos dictionary.

    One OMP step selects one frequency, meaning both the cosine and sine
    columns for that frequency are added to the support together.

    K means number of selected frequencies.
    """
    x = np.asarray(frame, dtype=np.float64)

    _, n_cols = dictionary.shape
    if n_cols % 2 != 0:
        raise ValueError("Expected sine/cos dictionary with an even number of columns.")

    n_freqs = n_cols // 2
    selected_freqs: list[int] = []
    residual = x.copy()

    k = min(k, n_freqs)
    x_norm = np.linalg.norm(x) + 1e-12

    for _ in range(k):
        corr = dictionary.T @ residual
        pair_scores = corr[0::2] ** 2 + corr[1::2] ** 2

        if selected_freqs:
            pair_scores[np.array(selected_freqs)] = -np.inf

        best_freq = int(np.argmax(pair_scores))
        if not np.isfinite(pair_scores[best_freq]):
            break

        selected_freqs.append(best_freq)

        selected_cols = []
        for fi in selected_freqs:
            selected_cols.extend([2 * fi, 2 * fi + 1])

        D_sel = dictionary[:, selected_cols]
        coeffs, *_ = np.linalg.lstsq(D_sel, x, rcond=None)
        residual = x - D_sel @ coeffs

        if np.linalg.norm(residual) / x_norm <= tol:
            break

    if not selected_freqs:
        return np.zeros_like(x)

    selected_cols = []
    for fi in selected_freqs:
        selected_cols.extend([2 * fi, 2 * fi + 1])

    D_sel = dictionary[:, selected_cols]
    coeffs, *_ = np.linalg.lstsq(D_sel, x, rcond=None)
    return D_sel @ coeffs


def denoise_signal(
    audio: np.ndarray,
    sample_rate: int,
    *,
    grid_type: str,
    n_freqs: int,
    f_min: float,
    f_max: float,
    K: int,
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
        dictionary, _ = build_sinusoidal_dictionary(
            sample_rate, window_size, freqs, normalize_columns=True
        )
        if dictionary_cache is not None:
            dictionary_cache[key] = dictionary

    output = np.zeros(n_samples)
    weight = np.zeros(n_samples)

    starts = list(range(0, max(n_samples - window_size + 1, 1), hop_size))
    if starts[-1] != n_samples - window_size:
        starts.append(n_samples - window_size)

    for start in starts:
        end = start + window_size
        frame = work[start:end]
        reconstructed = omp_reconstruct(frame, dictionary, k=K, tol=tol)

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
class OMPMethod(DenoisingMethod):
    params: dict[str, Any]
    dictionary_cache: dict = field(default_factory=dict)
    name: str = "omp"

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        dataset_name: str,
        overrides: dict[str, Any] | None = None,
    ) -> "OMPMethod":
        presets = config["methods"][cls.name]["parameter_presets"]
        params = dict(presets[dataset_name])
        if overrides:
            params.update(overrides)
        return cls(params=params)

    def denoise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return denoise_signal(
            audio,
            sample_rate,
            dictionary_cache=self.dictionary_cache,
            **self.params,
        )