"""
norm_method.py

Norm-based denoising method for audio/signal experiments.

This file implements an L1-regularized least-squares method solved by FISTA:

    min_c  0.5 ||A c - x||_2^2  +  lambda ||c||_1

where A is a sinusoidal dictionary. The L1 penalty encourages sparsity:
the solution uses only a small number of sinusoidal atoms to reconstruct
the signal, naturally suppressing noise spread across many frequencies.

Solver: FISTA (Fast Iterative Shrinkage-Thresholding Algorithm).
FISTA adds a Nesterov momentum step to plain ISTA, achieving O(1/k^2)
convergence versus ISTA's O(1/k) — a significant speedup at no extra cost.
"""

import time
from typing import Dict, Tuple

import numpy as np

from utils import compute_metrics


def build_sinusoidal_dictionary(
    num_samples: int,
    sample_rate: float,
    frequencies: np.ndarray,
    normalize_columns: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a real sinusoidal dictionary with cosine and sine columns.

    Parameters
    ----------
    num_samples : int
        Number of time samples N.
    sample_rate : float
        Sampling rate Fs in Hz.
    frequencies : np.ndarray
        Candidate frequencies in Hz.
    normalize_columns : bool
        If True, normalize each column to unit Euclidean norm.

    Returns
    -------
    A : np.ndarray
        Design matrix of shape (N, 2M), where M is the number of frequencies.
    column_norms : np.ndarray
        Norms used to scale each column (ones if normalize_columns is False).
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.ndim != 1 or len(frequencies) == 0:
        raise ValueError("frequencies must be a nonempty one-dimensional array.")

    n = np.arange(num_samples)
    columns = []

    for f in frequencies:
        columns.append(np.cos(2 * np.pi * f * n / sample_rate))
        columns.append(np.sin(2 * np.pi * f * n / sample_rate))

    A = np.column_stack(columns)

    column_norms = np.linalg.norm(A, axis=0)
    column_norms[column_norms == 0] = 1.0

    if normalize_columns:
        A = A / column_norms
    else:
        column_norms = np.ones_like(column_norms)

    return A, column_norms


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """
    Soft-thresholding (proximal operator for the L1 norm).

    S_lambda(z) = sign(z) * max(|z| - lambda, 0)
    """
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def l1_norm_denoise_fista(
    noisy_signal: np.ndarray,
    sample_rate: float,
    frequencies: np.ndarray,
    lambda_reg: float = 0.05,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Denoise using L1-regularized least squares solved by FISTA.

    Objective:
        min_c  0.5 ||A c - x||_2^2  +  lambda ||c||_1

    FISTA update rule (Beck & Teboulle, 2009):
        gradient  = A^T (A y - x)
        c_new     = S_{lambda/L}(y - gradient / L)
        t_new     = (1 + sqrt(1 + 4 t^2)) / 2
        y_new     = c_new + ((t - 1) / t_new) * (c_new - c_old)

    where L = ||A||_2^2 is the Lipschitz constant of the smooth gradient
    and S is the soft-threshold operator.

    Parameters
    ----------
    noisy_signal : np.ndarray
        One-dimensional noisy input signal.
    sample_rate : float
        Sampling rate in Hz.
    frequencies : np.ndarray
        Candidate sinusoidal frequencies in Hz.
    lambda_reg : float
        L1 regularization strength. Larger values yield sparser solutions.
    max_iter : int
        Maximum number of FISTA iterations.
    tol : float
        Stopping tolerance on the coefficient change ||c - c_old||.

    Returns
    -------
    denoised_signal : np.ndarray
        Reconstructed denoised signal.
    coefficients : np.ndarray
        Sparse coefficient vector.
    A : np.ndarray
        Sinusoidal dictionary used for reconstruction.
    num_iterations : int
        Number of FISTA iterations performed.
    """
    x = np.asarray(noisy_signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("noisy_signal must be one-dimensional.")
    if lambda_reg < 0:
        raise ValueError("lambda_reg must be nonnegative.")

    A, _ = build_sinusoidal_dictionary(len(x), sample_rate, frequencies, normalize_columns=True)

    # Lipschitz constant of gradient of 0.5 ||Ac - x||^2 is the spectral norm ||A||_2^2.
    lipschitz = np.linalg.norm(A, ord=2) ** 2 + 1e-12
    step_size = 1.0 / lipschitz

    c = np.zeros(A.shape[1])
    y = c.copy()           # momentum variable
    t = 1.0                # momentum scalar

    num_iterations = max_iter
    for iteration in range(1, max_iter + 1):
        c_old = c.copy()

        gradient = A.T @ (A @ y - x)
        c = soft_threshold(y - step_size * gradient, lambda_reg * step_size)

        # Nesterov momentum update
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = c + ((t - 1.0) / t_new) * (c - c_old)
        t = t_new

        if np.linalg.norm(c - c_old) < tol:
            num_iterations = iteration
            break

    denoised_signal = A @ c
    return denoised_signal, c, A, num_iterations


def run_norm_method(
    clean_signal: np.ndarray,
    noisy_signal: np.ndarray,
    sample_rate: float,
    frequencies: np.ndarray,
    lambda_reg: float = 0.05,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> Dict[str, object]:
    """
    Run L1 norm denoising (FISTA) and return denoised signal, metrics, and runtime.

    Parameters
    ----------
    clean_signal : np.ndarray
        Ground-truth clean signal.
    noisy_signal : np.ndarray
        Noisy observed signal.
    sample_rate : float
        Sampling rate in Hz.
    frequencies : np.ndarray
        Candidate sinusoidal frequencies in Hz.
    lambda_reg : float
        L1 regularization strength.
    max_iter : int
        Maximum FISTA iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    result : dict
        Dictionary containing denoised signal, metrics, coefficients,
        runtime, sparsity, and iteration count.
    """
    start_time = time.perf_counter()
    denoised_signal, coefficients, A, num_iterations = l1_norm_denoise_fista(
        noisy_signal=noisy_signal,
        sample_rate=sample_rate,
        frequencies=frequencies,
        lambda_reg=lambda_reg,
        max_iter=max_iter,
        tol=tol,
    )
    runtime_seconds = time.perf_counter() - start_time

    metrics = compute_metrics(clean_signal, noisy_signal, denoised_signal)

    nonzero_coefficients = int(np.sum(np.abs(coefficients) > 1e-8))
    condition_number = float(np.linalg.cond(A))

    return {
        "method": "L1 norm regularized least squares (FISTA)",
        "denoised_signal": denoised_signal,
        "coefficients": coefficients,
        "dictionary": A,
        "num_nonzero_coefficients": nonzero_coefficients,
        "lambda_reg": lambda_reg,
        "num_iterations": num_iterations,
        "condition_number": condition_number,
        "runtime_seconds": runtime_seconds,
        "metrics": metrics,
    }
