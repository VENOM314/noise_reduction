"""
fft_method.py

FFT-based denoising and compression methods for audio/signal experiments.

This file implements a simple frequency-domain baseline:
    1. Take the one-sided FFT (rfft) of the noisy signal.
    2. Keep only the largest frequency coefficients by magnitude.
    3. Set the rest to zero.
    4. Reconstruct the signal using the inverse rfft (irfft).

Using rfft instead of the full fft is correct for real-valued signals:
the full fft produces N coefficients, but the upper half are conjugate
mirrors of the lower half, giving only N//2+1 unique frequencies.
rfft exposes exactly these unique coefficients so keep_ratio applies to
the actual frequency content rather than wasting budget on redundant pairs.
"""

import time
from typing import Dict, Tuple

import numpy as np

from utils import compute_metrics


def fft_denoise_top_k(noisy_signal: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Denoise a signal by keeping the k largest one-sided FFT coefficients.

    Parameters
    ----------
    noisy_signal : np.ndarray
        One-dimensional noisy input signal.
    k : int
        Number of one-sided FFT coefficients to keep (out of N//2+1 unique ones).

    Returns
    -------
    denoised_signal : np.ndarray
        Reconstructed real-valued denoised signal.
    kept_indices : np.ndarray
        Indices (into the one-sided spectrum) of coefficients that were kept.
    """
    x = np.asarray(noisy_signal, dtype=float)
    n = len(x)

    if n == 0:
        raise ValueError("noisy_signal must not be empty.")
    if k <= 0:
        raise ValueError("k must be positive.")

    spectrum = np.fft.rfft(x)           # shape: (n//2 + 1,) — unique frequencies only
    k = min(k, len(spectrum))

    kept_indices = np.argsort(np.abs(spectrum))[-k:]

    filtered_spectrum = np.zeros_like(spectrum)
    filtered_spectrum[kept_indices] = spectrum[kept_indices]

    denoised_signal = np.fft.irfft(filtered_spectrum, n=n)

    return denoised_signal, kept_indices


def fft_denoise_keep_ratio(
    noisy_signal: np.ndarray, keep_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Denoise a signal by keeping a fixed fraction of the unique FFT coefficients.

    Parameters
    ----------
    noisy_signal : np.ndarray
        One-dimensional noisy input signal.
    keep_ratio : float
        Fraction of the one-sided spectrum to keep. Example: 0.1 keeps 10% of
        the N//2+1 unique frequency bins.

    Returns
    -------
    denoised_signal : np.ndarray
        Reconstructed real-valued denoised signal.
    kept_indices : np.ndarray
        Indices of one-sided FFT coefficients that were kept.
    """
    if not (0 < keep_ratio <= 1):
        raise ValueError("keep_ratio must satisfy 0 < keep_ratio <= 1.")

    n = len(noisy_signal)
    num_unique = n // 2 + 1             # number of unique one-sided frequency bins
    k = max(1, int(round(keep_ratio * num_unique)))
    return fft_denoise_top_k(noisy_signal, k)


def run_fft_method(
    clean_signal: np.ndarray,
    noisy_signal: np.ndarray,
    keep_ratio: float = 0.1,
) -> Dict[str, object]:
    """
    Run FFT thresholding and return denoised signal, metrics, and runtime.

    Parameters
    ----------
    clean_signal : np.ndarray
        Ground-truth clean signal.
    noisy_signal : np.ndarray
        Noisy observed signal.
    keep_ratio : float
        Fraction of unique FFT coefficients to keep.

    Returns
    -------
    result : dict
        Dictionary containing denoised signal, metrics, runtime, and kept indices.
    """
    start_time = time.perf_counter()
    denoised_signal, kept_indices = fft_denoise_keep_ratio(noisy_signal, keep_ratio=keep_ratio)
    runtime_seconds = time.perf_counter() - start_time

    metrics = compute_metrics(clean_signal, noisy_signal, denoised_signal)

    return {
        "method": "FFT thresholding",
        "denoised_signal": denoised_signal,
        "kept_indices": kept_indices,
        "num_kept_coefficients": len(kept_indices),
        "keep_ratio": keep_ratio,
        "runtime_seconds": runtime_seconds,
        "metrics": metrics,
    }
