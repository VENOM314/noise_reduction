from __future__ import annotations

import numpy as np
from scipy.signal import stft


def zero_mean_unit_norm(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    audio = audio - np.mean(audio)
    norm = np.linalg.norm(audio)
    if norm < 1e-12:
        return np.zeros_like(audio)
    return audio / norm


def scale_to_reference(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    denom = np.dot(estimate, estimate)
    if denom < 1e-12:
        return estimate
    return estimate * (np.dot(reference, estimate) / denom)


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    estimate = scale_to_reference(reference, estimate)
    signal_power = np.mean(reference**2)
    error_power = np.mean((reference - estimate) ** 2)
    if error_power < 1e-12:
        return float("inf")
    return float(10 * np.log10((signal_power + 1e-12) / error_power))


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(zero_mean_unit_norm(x), zero_mean_unit_norm(y)))


def spectrogram_corr(reference: np.ndarray, estimate: np.ndarray, sample_rate: int) -> float:
    _, _, ref_spec = stft(reference, fs=sample_rate, nperseg=1024, noverlap=768)
    _, _, est_spec = stft(estimate, fs=sample_rate, nperseg=1024, noverlap=768)
    return pearson_corr(np.log1p(np.abs(ref_spec)).ravel(), np.log1p(np.abs(est_spec)).ravel())
