from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from noise_cancellation.audio import peak_normalize


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio**2)) + 1e-12)


def scale_noise_to_ratio(clean: np.ndarray, noise: np.ndarray, ratio: float) -> np.ndarray:
    return noise * (ratio * rms(clean) / rms(noise))


def prevent_clipping(noisy: np.ndarray, noise: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    peak = float(np.max(np.abs(noisy))) if noisy.size else 0.0
    if peak <= 1.0:
        return noisy, noise
    noisy = noisy / peak * 0.95
    if noise is not None:
        noise = noise / peak * 0.95
    return noisy, noise


def slice_or_tile(audio: np.ndarray, start: int, length: int) -> np.ndarray:
    if len(audio) < length:
        audio = np.tile(audio, math.ceil(length / len(audio)))
    if start + length > len(audio):
        start = 0
    return audio[start : start + length]


def apply_distant_hall_effect(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    lowpass = butter(4, 3200, btype="lowpass", fs=sample_rate, output="sos")
    highpass = butter(2, 180, btype="highpass", fs=sample_rate, output="sos")
    filtered = sosfilt(highpass, sosfilt(lowpass, audio))

    pre_delay = int(0.055 * sample_rate)
    decay_len = int(0.85 * sample_rate)
    t = np.arange(decay_len) / sample_rate
    rng = np.random.default_rng(42)
    impulse = rng.normal(0.0, 1.0, decay_len) * np.exp(-t * 4.5)
    impulse[:pre_delay] = 0.0
    impulse[pre_delay] = 1.0
    impulse = impulse / (np.max(np.abs(impulse)) + 1e-12)

    reverbed = fftconvolve(filtered, impulse, mode="full")[: len(filtered)]
    return peak_normalize(0.35 * filtered + 0.65 * reverbed, peak=0.8)
