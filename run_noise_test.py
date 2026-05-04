"""
run_noise_test.py

Test FFT and norm denoising against real noise sources.

Mixes a clean reference audio file with a noise file at a target SNR,
runs both denoising methods, and reports full metrics (MSE, RMSE, SNR
improvement) — possible because we have the clean reference.

Usage
-----
    # Test one clean file against one noise file:
    python run_noise_test.py --clean path/to/clean.wav --noise path/to/noise.wav

    # Batch: test all 3 noise types against one experiment folder:
    python run_noise_test.py --experiment real_world_selected
    python run_noise_test.py --experiment melody_music
    python run_noise_test.py --experiment synthetic_piano_notes

Data folder is assumed to be at:
    C:\\Users\\Chris\\OneDrive\\Desktop\\data
"""

import argparse
import os

import numpy as np

from fft_method import run_fft_method
from norm_method import run_norm_method
from utils import load_audio, save_audio

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

DATA_DIR = r"C:\Users\Chris\OneDrive\Desktop\data"

NOISE_FILES = {
    "background_speech":    "background_speech.wav",
    "airplane":             "transportation_airplane.wav",
    "metro":                "transportation_tmetro.wav",
}


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, target_snr_db: float) -> np.ndarray:
    """
    Mix clean signal with noise trimmed/looped to match length, scaled to
    achieve target_snr_db input SNR.

    SNR = 10 log10(power_clean / power_noise)
    """
    if len(noise) < len(clean):
        repeats = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[:len(clean)]

    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return clean.copy()

    target_noise_power = clean_power / (10 ** (target_snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    return clean + scale * noise


def run_one_test(
    clean_path: str,
    noise_path: str,
    target_snr_db: float = 5.0,
    keep_ratio: float = 0.05,
    lambda_reg: float = 0.05,
    num_frequencies: int = 100,
    save_outputs: bool = False,
    plot: bool = False,
    label: str = "",
) -> dict:
    """
    Mix, denoise, and return metrics for one clean+noise pair.
    """
    sr_c, clean = load_audio(clean_path)
    sr_n, noise = load_audio(noise_path)

    if sr_n != sr_c:
        from scipy.signal import resample
        noise = resample(noise, int(len(noise) * sr_c / sr_n))

    noisy = mix_at_snr(clean, noise, target_snr_db)

    fft_result = run_fft_method(
        clean_signal=clean,
        noisy_signal=noisy,
        keep_ratio=keep_ratio,
    )

    frequencies = np.linspace(20, sr_c / 2, num_frequencies)
    norm_result = run_norm_method(
        clean_signal=clean,
        noisy_signal=noisy,
        sample_rate=sr_c,
        frequencies=frequencies,
        lambda_reg=lambda_reg,
        max_iter=1000,
        tol=1e-6,
    )

    if save_outputs:
        base = os.path.splitext(clean_path)[0]
        noise_tag = os.path.splitext(os.path.basename(noise_path))[0]
        save_audio(base + f"_noisy_{noise_tag}.wav", noisy, sr_c)
        save_audio(base + f"_fft_denoised_{noise_tag}.wav", fft_result["denoised_signal"], sr_c)
        save_audio(base + f"_norm_denoised_{noise_tag}.wav", norm_result["denoised_signal"], sr_c)

    if plot and HAS_MATPLOTLIB:
        t = np.arange(len(clean)) / sr_c
        plt.figure(figsize=(12, 5))
        plt.plot(t, clean,  label="Clean",        linewidth=1.5)
        plt.plot(t, noisy,  label="Noisy input",  alpha=0.45)
        plt.plot(t, fft_result["denoised_signal"],  label="FFT denoised")
        plt.plot(t, norm_result["denoised_signal"], label="L1 norm denoised")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(label or os.path.basename(clean_path))
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "fft":  fft_result,
        "norm": norm_result,
    }


def print_row(label: str, fft_result: dict, norm_result: dict) -> None:
    fm = fft_result["metrics"]
    nm = norm_result["metrics"]
    print(
        f"  {label:<30s}"
        f"  FFT  SNR_in={fm['snr_before_db']:+.1f}dB  SNR_out={fm['snr_after_db']:+.1f}dB  "
        f"gain={fm['snr_improvement_db']:+.1f}dB  t={fft_result['runtime_seconds']:.3f}s"
    )
    print(
        f"  {'':30s}"
        f"  Norm SNR_in={nm['snr_before_db']:+.1f}dB  SNR_out={nm['snr_after_db']:+.1f}dB  "
        f"gain={nm['snr_improvement_db']:+.1f}dB  t={norm_result['runtime_seconds']:.3f}s  "
        f"iters={norm_result['num_iterations']}  cond={norm_result['condition_number']:.2f}"
    )


def run_batch(experiment: str, target_snr_db: float = 5.0, num_files: int = 5) -> None:
    exp_dir = os.path.join(DATA_DIR, "experiments", experiment)
    clean_dir = os.path.join(exp_dir, "clean")
    noise_dir = os.path.join(exp_dir, "noise")

    clean_files = sorted(f for f in os.listdir(clean_dir) if f.endswith(".wav"))[:num_files]

    print(f"\nExperiment : {experiment}")
    print(f"Input SNR  : {target_snr_db} dB")
    print(f"Clean files: {len(clean_files)}")
    print("=" * 90)

    for noise_label, noise_filename in NOISE_FILES.items():
        noise_path = os.path.join(noise_dir, noise_filename)
        print(f"\nNoise: {noise_label}")
        print("-" * 90)

        for cf in clean_files:
            clean_path = os.path.join(clean_dir, cf)
            results = run_one_test(
                clean_path=clean_path,
                noise_path=noise_path,
                target_snr_db=target_snr_db,
            )
            print_row(cf, results["fft"], results["norm"])

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test denoising against real noise sources.")
    parser.add_argument("--clean",      type=str, help="Path to clean WAV file.")
    parser.add_argument("--noise",      type=str, help="Path to noise WAV file.")
    parser.add_argument("--experiment", type=str,
                        choices=["real_world_selected", "melody_music", "synthetic_piano_notes"],
                        help="Batch-test an experiment folder.")
    parser.add_argument("--snr",          type=float, default=5.0,
                        help="Target input SNR in dB (default: 5).")
    parser.add_argument("--keep_ratio",   type=float, default=0.05,
                        help="Fraction of FFT bins to keep (default: 0.05).")
    parser.add_argument("--lambda_reg",   type=float, default=0.05,
                        help="L1 regularization strength for norm method (default: 0.05).")
    parser.add_argument("--num_files",    type=int,   default=5,
                        help="Number of clean files to test in batch mode (default: 5).")
    parser.add_argument("--save",         action="store_true",
                        help="Save noisy and denoised WAV files next to clean file.")
    parser.add_argument("--plot",         action="store_true",
                        help="Show waveform plot for each test.")
    args = parser.parse_args()

    if args.experiment:
        run_batch(args.experiment, target_snr_db=args.snr, num_files=args.num_files)

    elif args.clean and args.noise:
        results = run_one_test(
            clean_path=args.clean,
            noise_path=args.noise,
            target_snr_db=args.snr,
            keep_ratio=args.keep_ratio,
            lambda_reg=args.lambda_reg,
            save_outputs=args.save,
            plot=args.plot,
            label=f"{os.path.basename(args.clean)}  +  {os.path.basename(args.noise)}",
        )
        print()
        print_row(
            f"{os.path.basename(args.clean)} + {os.path.basename(args.noise)}",
            results["fft"],
            results["norm"],
        )
        print()

    else:
        parser.print_help()
