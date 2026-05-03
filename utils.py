"""
utils.py

Shared evaluation utilities for all denoising methods.
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np


def load_audio(filepath: str) -> Tuple[int, np.ndarray]:
    """
    Load a WAV or MP3 file and return the sample rate and signal as float64 mono.

    Handles 16-bit, 32-bit integer, and float WAV files, and MP3 files.
    Stereo files are converted to mono by averaging channels.

    Requirements
    ------------
    WAV : scipy
    MP3 : pydub  (pip install pydub)  +  ffmpeg  (conda install -c conda-forge ffmpeg)

    Parameters
    ----------
    filepath : str
        Path to the audio file (.wav or .mp3).

    Returns
    -------
    sample_rate : int
        Sampling rate in Hz.
    signal : np.ndarray
        Mono signal normalized to the range [-1.0, 1.0].
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".mp3":
        # If ffmpeg is not on PATH, look for the WinGet installation and add it.
        import shutil as _shutil
        if _shutil.which("ffmpeg") is None:
            import glob as _glob
            winget_dir = os.path.join(
                os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages"
            )
            matches = _glob.glob(
                os.path.join(winget_dir, "**", "bin", "ffmpeg.exe"), recursive=True
            )
            if matches:
                os.environ["PATH"] = os.path.dirname(matches[0]) + os.pathsep + os.environ["PATH"]

        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError("Install pydub to load MP3 files: pip install pydub")

        audio = AudioSegment.from_mp3(filepath).set_channels(1)
        sample_rate = audio.frame_rate
        samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
        max_val = float(1 << (audio.sample_width * 8 - 1))
        signal = samples / max_val

    elif ext == ".wav":
        from scipy.io import wavfile

        sample_rate, data = wavfile.read(filepath)

        if data.dtype == np.int16:
            signal = data.astype(np.float64) / 32768.0
        elif data.dtype == np.int32:
            signal = data.astype(np.float64) / 2147483648.0
        else:
            signal = data.astype(np.float64)

        if signal.ndim == 2:
            signal = signal.mean(axis=1)

    else:
        raise ValueError(f"Unsupported file format: '{ext}'. Supported: .wav, .mp3")

    return sample_rate, signal


def save_audio(filepath: str, signal: np.ndarray, sample_rate: int) -> None:
    """
    Save a float64 signal as a 16-bit WAV file.

    Parameters
    ----------
    filepath : str
        Output path (should end in .wav).
    signal : np.ndarray
        Signal to save. Values are clipped to [-1.0, 1.0] before writing.
    sample_rate : int
        Sampling rate in Hz.
    """
    from scipy.io import wavfile

    clipped = np.clip(signal, -1.0, 1.0)
    wavfile.write(filepath, sample_rate, (clipped * 32767).astype(np.int16))


def compute_metrics(
    clean_signal: Optional[np.ndarray],
    noisy_signal: np.ndarray,
    denoised_signal: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standard denoising evaluation metrics.

    If clean_signal is None (real audio with no ground truth), returns an
    empty dict — the denoising still runs, metrics just cannot be computed.

    Parameters
    ----------
    clean_signal : np.ndarray or None
        Ground-truth clean signal, or None if unavailable.
    noisy_signal : np.ndarray
        Noisy observed signal.
    denoised_signal : np.ndarray
        Output signal after denoising.

    Returns
    -------
    metrics : dict
        MSE, RMSE, SNR before/after denoising, and SNR improvement in dB.
        Empty dict if clean_signal is None.
    """
    if clean_signal is None:
        return {}

    clean = np.asarray(clean_signal, dtype=float)
    noisy = np.asarray(noisy_signal, dtype=float)
    denoised = np.asarray(denoised_signal, dtype=float)

    if clean.shape != noisy.shape or clean.shape != denoised.shape:
        raise ValueError(
            "clean_signal, noisy_signal, and denoised_signal must have the same shape."
        )

    noise_before = noisy - clean
    error_after = denoised - clean

    mse_noisy = float(np.mean(noise_before ** 2))
    mse_denoised = float(np.mean(error_after ** 2))

    rmse_noisy = float(np.sqrt(mse_noisy))
    rmse_denoised = float(np.sqrt(mse_denoised))

    signal_power = float(np.mean(clean ** 2))

    snr_before = (
        float(10 * np.log10(signal_power / mse_noisy)) if mse_noisy > 0 else float("inf")
    )
    snr_after = (
        float(10 * np.log10(signal_power / mse_denoised)) if mse_denoised > 0 else float("inf")
    )

    return {
        "mse_noisy": mse_noisy,
        "mse_denoised": mse_denoised,
        "rmse_noisy": rmse_noisy,
        "rmse_denoised": rmse_denoised,
        "snr_before_db": snr_before,
        "snr_after_db": snr_after,
        "snr_improvement_db": snr_after - snr_before,
    }
