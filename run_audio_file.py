"""
run_audio_file.py

Run FFT and norm denoising on a real noisy audio file.

Usage
-----
    python run_audio_file.py path/to/noisy_audio.wav

The script will:
    1. Load the WAV file (mono or stereo, any standard bit depth).
    2. Run FFT thresholding and L1 norm (FISTA) denoising.
    3. Save the two denoised outputs next to the input file.
    4. Print a summary and show a waveform comparison plot.

No clean reference is needed — this works directly on noisy recordings.
"""

import os
import sys

import numpy as np

from fft_method import run_fft_method
from norm_method import run_norm_method
from utils import load_audio, save_audio

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def run_on_audio_file(
    filepath: str,
    keep_ratio: float = 0.05,
    lambda_reg: float = 0.05,
    num_frequencies: int = 50,
    max_norm_seconds: float = 1.0,
) -> None:
    """
    Denoise a WAV or MP3 file with both methods and save the results.

    The FFT method runs on the full file. The norm method (FISTA) runs on the
    first max_norm_seconds of audio — its dictionary matrix is O(N × 2M) and
    becomes too large for long files.

    Parameters
    ----------
    filepath : str
        Path to the noisy input audio file (.wav or .mp3).
    keep_ratio : float
        Fraction of FFT coefficients to keep (FFT method).
    lambda_reg : float
        L1 regularization strength (norm method).
    num_frequencies : int
        Number of candidate frequencies in the sinusoidal dictionary.
    max_norm_seconds : float
        Maximum audio length (seconds) for the norm method.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    print(f"Loading: {filepath}")
    sample_rate, noisy_signal = load_audio(filepath)
    duration = len(noisy_signal) / sample_rate
    print(f"  Sample rate : {sample_rate} Hz")
    print(f"  Duration    : {duration:.2f} seconds")
    print(f"  Samples     : {len(noisy_signal)}")

    # --- FFT method (full file) ---
    print("\nRunning FFT thresholding (full file)...")
    fft_result = run_fft_method(
        clean_signal=None,
        noisy_signal=noisy_signal,
        keep_ratio=keep_ratio,
    )
    print(f"  Runtime            : {fft_result['runtime_seconds']:.4f} s")
    print(f"  Kept coefficients  : {fft_result['num_kept_coefficients']}")

    # --- Norm method (excerpt only) ---
    max_norm_samples = int(max_norm_seconds * sample_rate)
    norm_signal = noisy_signal[:max_norm_samples]
    norm_duration = len(norm_signal) / sample_rate
    if len(noisy_signal) > max_norm_samples:
        print(f"\nRunning L1 norm (FISTA) on first {norm_duration:.1f}s excerpt...")
        print(f"  (Dictionary matrix is O(N × 2M); full file would require too much RAM.)")
    else:
        print("\nRunning L1 norm (FISTA)...")

    frequencies = np.linspace(20, sample_rate / 2, num_frequencies)
    norm_result = run_norm_method(
        clean_signal=None,
        noisy_signal=norm_signal,
        sample_rate=sample_rate,
        frequencies=frequencies,
        lambda_reg=lambda_reg,
        max_iter=1000,
        tol=1e-6,
    )
    print(f"  Runtime            : {norm_result['runtime_seconds']:.4f} s")
    print(f"  Nonzero components : {norm_result['num_nonzero_coefficients']}")
    print(f"  FISTA iterations   : {norm_result['num_iterations']}")
    print(f"  Condition number   : {norm_result['condition_number']:.4f}")

    # --- Save outputs ---
    base = os.path.splitext(filepath)[0]
    fft_out = base + "_fft_denoised.wav"
    norm_out = base + f"_norm_denoised_{norm_duration:.1f}s.wav"

    save_audio(fft_out, fft_result["denoised_signal"], sample_rate)
    save_audio(norm_out, norm_result["denoised_signal"], sample_rate)

    print(f"\nSaved: {fft_out}")
    print(f"Saved: {norm_out}")

    # --- Plot (show the excerpt window for both methods) ---
    if HAS_MATPLOTLIB:
        n_plot = len(norm_signal)
        t = np.arange(n_plot) / sample_rate
        plt.figure(figsize=(12, 6))
        plt.plot(t, norm_signal, label="Noisy input", alpha=0.45)
        plt.plot(t, fft_result["denoised_signal"][:n_plot], label="FFT denoised")
        plt.plot(t, norm_result["denoised_signal"], label="L1 norm denoised")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.title(f"Denoising comparison (first {norm_duration:.1f}s): {os.path.basename(filepath)}")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_audio_file.py path/to/noisy_audio.wav")
        sys.exit(1)

    run_on_audio_file(filepath=sys.argv[1])
