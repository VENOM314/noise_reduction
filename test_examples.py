import numpy as np

from fft_method import run_fft_method
from norm_method import run_norm_method


try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def make_test_signal(case_name="three_sinusoids", sample_rate=1000, duration=1.0, noise_level=0.5, seed=0):
    """
    Generate clean and noisy test signals.

    Parameters
    ----------
    case_name : str
        Test case name. Options:
            "three_sinusoids"
            "low_frequency_drift"
            "impulse_noise"
            "mixed_noise"
    sample_rate : float
        Sampling rate in Hz.
    duration : float
        Signal duration in seconds.
    noise_level : float
        Noise strength.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    t : np.ndarray
        Time axis.
    clean_signal : np.ndarray
        Clean signal.
    noisy_signal : np.ndarray
        Noisy signal.
    """
    rng = np.random.default_rng(seed)
    n = int(sample_rate * duration)
    t = np.arange(n) / sample_rate

    if case_name == "three_sinusoids":
        clean_signal = (
            np.sin(2 * np.pi * 50 * t)
            + 0.5 * np.sin(2 * np.pi * 120 * t)
            + 0.3 * np.sin(2 * np.pi * 250 * t)
        )
        noise = noise_level * rng.normal(size=n)

    elif case_name == "low_frequency_drift":
        clean_signal = (
            np.sin(2 * np.pi * 70 * t)
            + 0.6 * np.sin(2 * np.pi * 180 * t)
        )
        drift = 0.8 * np.sin(2 * np.pi * 5 * t)
        white_noise = noise_level * rng.normal(size=n)
        noise = drift + white_noise

    elif case_name == "impulse_noise":
        clean_signal = (
            np.sin(2 * np.pi * 60 * t)
            + 0.7 * np.sin(2 * np.pi * 130 * t)
        )
        noise = 0.25 * rng.normal(size=n)
        impulse_indices = rng.choice(n, size=max(1, n // 50), replace=False)
        noise[impulse_indices] += rng.choice([-3.0, 3.0], size=len(impulse_indices))

    elif case_name == "mixed_noise":
        clean_signal = (
            np.sin(2 * np.pi * 40 * t)
            + 0.5 * np.sin(2 * np.pi * 90 * t)
            + 0.35 * np.sin(2 * np.pi * 220 * t)
        )
        white_noise = noise_level * rng.normal(size=n)
        interference = 0.6 * np.sin(2 * np.pi * 300 * t)
        noise = white_noise + interference

    else:
        raise ValueError(f"Unknown case_name: {case_name}")

    noisy_signal = clean_signal + noise
    return t, clean_signal, noisy_signal


def print_metrics(result):
    print(f"\nMethod: {result['method']}")
    print(f"Runtime: {result['runtime_seconds']:.6f} seconds")

    if "num_kept_coefficients" in result:
        print(f"Number of kept FFT coefficients: {result['num_kept_coefficients']}")

    if "num_nonzero_coefficients" in result:
        print(f"Number of nonzero L1 coefficients: {result['num_nonzero_coefficients']}")
        print(f"Number of FISTA iterations: {result['num_iterations']}")
        print(f"Dictionary condition number: {result['condition_number']:.4f}")

    for key, value in result["metrics"].items():
        print(f"{key}: {value:.6f}")


def run_one_example(case_name, sample_rate=1000, duration=1.0):
    print("=" * 80)
    print(f"Test case: {case_name}")

    t, clean_signal, noisy_signal = make_test_signal(
        case_name=case_name,
        sample_rate=sample_rate,
        duration=duration,
        noise_level=0.5,
        seed=0,
    )

    # FFT method: keep 5% of FFT coefficients.
    fft_result = run_fft_method(
        clean_signal=clean_signal,
        noisy_signal=noisy_signal,
        keep_ratio=0.05,
    )

    # Norm method: use candidate frequencies from 20 Hz to Nyquist frequency.
    # You can increase number of frequencies for a denser dictionary,
    # but it will make the norm method slower.
    frequencies = np.linspace(20, sample_rate / 2, 250)

    norm_result = run_norm_method(
        clean_signal=clean_signal,
        noisy_signal=noisy_signal,
        sample_rate=sample_rate,
        frequencies=frequencies,
        lambda_reg=0.05,
        max_iter=1000,
        tol=1e-6,
    )

    print_metrics(fft_result)
    print_metrics(norm_result)

    if HAS_MATPLOTLIB:
        plt.figure(figsize=(12, 6))
        plt.plot(t, clean_signal, label="Clean signal", linewidth=2)
        plt.plot(t, noisy_signal, label="Noisy signal", alpha=0.45)
        plt.plot(t, fft_result["denoised_signal"], label="FFT denoised")
        plt.plot(t, norm_result["denoised_signal"], label="L1 norm denoised")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.title(f"Denoising comparison: {case_name}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return fft_result, norm_result


if __name__ == "__main__":
    test_cases = [
        "three_sinusoids",
        "low_frequency_drift",
        "impulse_noise",
        "mixed_noise",
    ]

    for case in test_cases:
        run_one_example(case_name=case, sample_rate=1000, duration=1.0)
