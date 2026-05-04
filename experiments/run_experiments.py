from __future__ import annotations

import argparse
import csv
import shutil
import stat
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from noise_cancellation.audio import load_mono, resample_to, save_wav
from noise_cancellation.config import load_config, output_root
from noise_cancellation.data_prep.datasets import DATASET_NAMES, normalize_dataset_name
from noise_cancellation.methods import available_methods, create_method
from noise_cancellation.methods.base import DenoisingMethod
from noise_cancellation.metrics import snr_db, spectrogram_corr
from noise_cancellation.plots import write_grouped_bar_svg, write_metric_svg, write_summary_bar_svg


DATA_ROOT = PROJECT_ROOT / "data"
CLEAN_ROOT = DATA_ROOT / "clean"
NOISY_ROOT = DATA_ROOT / "noisy"


def reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, onerror=retry_readonly_remove)
    path.mkdir(parents=True, exist_ok=True)


def retry_readonly_remove(function, path: str, _exc_info) -> None:
    Path(path).chmod(stat.S_IWRITE)
    function(path)


def dataset_names(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(config.get("datasets", DATASET_NAMES))


def experiment_dirs(method_name: str, dataset_name: str) -> dict[str, Path]:
    base_output = output_root(PROJECT_ROOT) / method_name / dataset_name
    return {
        "clean": CLEAN_ROOT / dataset_name,
        "noisy": NOISY_ROOT / dataset_name,
        "denoised": base_output / "denoised",
        "metrics": base_output / "metrics",
        "plots": base_output / "plots",
    }


def ensure_prepared_dataset(dataset_name: str, dirs: dict[str, Path]) -> None:
    clean_files = sorted(dirs["clean"].glob("*.wav"))
    noisy_files = sorted(dirs["noisy"].rglob("*_noisy.wav"))
    if not clean_files or not noisy_files:
        raise FileNotFoundError(
            f"Prepared data for '{dataset_name}' was not found. "
            "Run experiments/prepare_data.py first or use the VS Code 'Prepare Data' launch config."
        )


def parse_noisy_stem(stem: str, noise_types: list[str]) -> tuple[str, str] | None:
    if stem.endswith("_denoised"):
        stem = stem[: -len("_denoised")]
    if not stem.endswith("_noisy"):
        return None
    without_suffix = stem[: -len("_noisy")]
    for noise_type in sorted(noise_types, key=len, reverse=True):
        suffix = f"_{noise_type}"
        if without_suffix.endswith(suffix):
            return without_suffix[: -len(suffix)], noise_type
    return None


def collect_noisy_paths(noisy_root: Path, limit: int | None = None) -> list[Path]:
    paths = sorted(noisy_root.rglob("*_noisy.wav"))
    if limit is not None:
        return paths[:limit]
    return paths


def denoise_dataset(
    method: DenoisingMethod,
    noisy_paths: list[Path],
    denoised_root: Path,
    noisy_root: Path,
    sample_rate: int,
) -> list[Path]:
    reset_output_dir(denoised_root)
    saved = []
    for index, noisy_path in enumerate(noisy_paths, start=1):
        rel = noisy_path.relative_to(noisy_root)
        save_path = denoised_root / rel.parent / f"{noisy_path.stem}_denoised.wav"
        audio, source_rate = load_mono(noisy_path)
        audio = resample_to(audio, source_rate, sample_rate)
        denoised = method.denoise(audio, sample_rate)
        save_wav(save_path, denoised, sample_rate)
        saved.append(save_path)
        if index % 10 == 0:
            print(f"  denoised {index}/{len(noisy_paths)}")
    return saved


def evaluate_dataset(
    clean_dir: Path,
    noisy_root: Path,
    denoised_root: Path,
    metrics_dir: Path,
    plots_dir: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_rate = int(config["sample_rate"])
    noise_types = config["active_noise_types"]
    denoised_index = {}
    for denoised_path in sorted(denoised_root.rglob("*_denoised.wav")):
        parsed = parse_noisy_stem(denoised_path.stem, noise_types)
        if parsed:
            denoised_index[parsed] = denoised_path

    rows = []
    for noisy_path in sorted(noisy_root.rglob("*_noisy.wav")):
        parsed = parse_noisy_stem(noisy_path.stem, noise_types)
        if not parsed or parsed not in denoised_index:
            continue
        clean_stem, noise_type = parsed
        clean_path = clean_dir / f"{clean_stem}.wav"
        if not clean_path.exists():
            continue

        clean, clean_rate = load_mono(clean_path)
        noisy, noisy_rate = load_mono(noisy_path)
        denoised, denoised_rate = load_mono(denoised_index[parsed])
        clean = resample_to(clean, clean_rate, sample_rate)
        noisy = resample_to(noisy, noisy_rate, sample_rate)
        denoised = resample_to(denoised, denoised_rate, sample_rate)
        length = min(len(clean), len(noisy), len(denoised))
        clean, noisy, denoised = clean[:length], noisy[:length], denoised[:length]

        snr_noisy = snr_db(clean, noisy)
        snr_denoised = snr_db(clean, denoised)
        rows.append(
            {
                "file": clean_stem,
                "noise_type": noise_type,
                "snr_noisy": snr_noisy,
                "snr_denoised": snr_denoised,
                "snr_improvement": snr_denoised - snr_noisy,
                "spectrogram_corr_noisy": spectrogram_corr(clean, noisy, sample_rate),
                "spectrogram_corr_denoised": spectrogram_corr(clean, denoised, sample_rate),
            }
        )

    metric_fields = [
        "snr_noisy",
        "snr_denoised",
        "snr_improvement",
        "spectrogram_corr_noisy",
        "spectrogram_corr_denoised",
    ]
    summary_rows = summarize_rows(rows, ["noise_type"], metric_fields)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = metrics_dir / "denoising_metrics.csv"
    summary_csv = metrics_dir / "denoising_metrics_summary.csv"
    write_csv(detail_csv, rows, ["file", "noise_type", *metric_fields])
    write_csv(summary_csv, summary_rows, ["noise_type", "count", *metric_fields])
    write_summary_bar_svg(summary_csv, plots_dir / "snr_improvement_by_noise.svg", "snr_improvement", "SNR improvement by noise")
    write_grouped_bar_svg(
        summary_csv,
        plots_dir / "spectrogram_corr_by_noise.svg",
        ("spectrogram_corr_noisy", "spectrogram_corr_denoised"),
        ("Noisy", "Denoised"),
        "Spectrogram correlation by noise",
        "spectrogram correlation",
    )
    return rows, summary_rows


def summarize_rows(rows: list[dict[str, Any]], keys: list[str], metric_fields: list[str]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    summary_rows = []
    for key_values, group_rows in sorted(grouped.items()):
        summary = {key: value for key, value in zip(keys, key_values)}
        summary["count"] = len(group_rows)
        for field in metric_fields:
            summary[field] = float(np.mean([row[field] for row in group_rows]))
        summary_rows.append(summary)
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_dataset(method_name: str, dataset_name: str, config: dict[str, Any], limit: int | None = None) -> None:
    dirs = experiment_dirs(method_name, dataset_name)
    ensure_prepared_dataset(dataset_name, dirs)
    noisy_paths = collect_noisy_paths(dirs["noisy"], limit=limit)
    method = create_method(method_name, config, dataset_name)
    print(f"Running {method_name} on {dataset_name}: {len(noisy_paths)} noisy files")
    denoise_dataset(method, noisy_paths, dirs["denoised"], dirs["noisy"], int(config["sample_rate"]))
    _, summary_rows = evaluate_dataset(dirs["clean"], dirs["noisy"], dirs["denoised"], dirs["metrics"], dirs["plots"], config)
    print_summary(dataset_name, summary_rows)


def print_summary(dataset_name: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{dataset_name} summary")
    for row in rows:
        print(
            f"{row['noise_type']:>18} "
            f"n={row['count']:>3} "
            f"snr {row['snr_noisy']:.2f}->{row['snr_denoised']:.2f} "
            f"spec {row['spectrogram_corr_noisy']:.3f}->{row['spectrogram_corr_denoised']:.3f}"
        )


def run_parameter_sweep(method_name: str, dataset_name: str, config: dict[str, Any]) -> None:
    if method_name != "least_squares":
        raise ValueError("Parameter sweep is currently defined only for least_squares.")
    dirs = experiment_dirs(method_name, dataset_name)
    ensure_prepared_dataset(dataset_name, dirs)
    sweep_config = config["methods"][method_name]["parameter_sweep"]
    sample_rate = int(config["sample_rate"])
    clean_limit = int(sweep_config["clean_clip_limit"])
    clean_stems = {path.stem for path in sorted(dirs["clean"].glob("*.wav"))[:clean_limit]}
    metric_fields = ["snr_improvement", "spectrogram_corr_denoised"]
    aggregate_rows = []

    for k_value in sweep_config["K_values"]:
        overrides = {
            "K": int(k_value),
            "n_freqs": int(sweep_config["n_freqs"]),
            "window_size": int(sweep_config["window_size"]),
            "hop_size": int(sweep_config["hop_size"]),
        }
        method = create_method(method_name, config, dataset_name, overrides=overrides)
        rows = []
        for noisy_path in collect_noisy_paths(dirs["noisy"]):
            parsed = parse_noisy_stem(noisy_path.stem, config["active_noise_types"])
            if parsed is None:
                continue
            clean_stem, noise_type = parsed
            if clean_stem not in clean_stems:
                continue
            clean_path = dirs["clean"] / f"{clean_stem}.wav"
            clean, clean_rate = load_mono(clean_path)
            noisy, noisy_rate = load_mono(noisy_path)
            clean = resample_to(clean, clean_rate, sample_rate)
            noisy = resample_to(noisy, noisy_rate, sample_rate)
            denoised = method.denoise(noisy, sample_rate)
            length = min(len(clean), len(noisy), len(denoised))
            clean, noisy, denoised = clean[:length], noisy[:length], denoised[:length]
            snr_noisy = snr_db(clean, noisy)
            snr_denoised = snr_db(clean, denoised)
            rows.append(
                {
                    "K": k_value,
                    "noise_type": noise_type,
                    "snr_improvement": snr_denoised - snr_noisy,
                    "spectrogram_corr_denoised": spectrogram_corr(clean, denoised, sample_rate),
                }
            )

        aggregate_rows.extend(summarize_rows(rows, ["K", "noise_type"], metric_fields))

    sweep_dir = output_root(PROJECT_ROOT) / method_name / "parameter_sweep"
    csv_path = sweep_dir / f"{dataset_name}_sweep_metrics.csv"
    write_csv(csv_path, aggregate_rows, ["K", "noise_type", "count", *metric_fields])
    for metric in metric_fields:
        write_metric_svg(
            csv_path,
            sweep_dir / "plots" / f"{dataset_name}_{metric}_by_K.svg",
            metric,
            f"{method_name} {dataset_name}: {metric} by K",
        )
    print(f"Wrote sweep metrics: {csv_path}")


def main() -> None:
    config = load_config(PROJECT_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=available_methods(), default="least_squares")
    parser.add_argument("--dataset", choices=("all", *dataset_names(config)), default="all")
    parser.add_argument("--parameter-sweep", action="store_true", help="run the method-specific parameter sweep")
    parser.add_argument("--limit", type=int, default=None, help="optional noisy-file limit for quicker runs")
    args = parser.parse_args()

    selected_datasets = dataset_names(config) if args.dataset == "all" else (normalize_dataset_name(args.dataset),)
    for dataset_name in selected_datasets:
        if args.parameter_sweep:
            run_parameter_sweep(args.method, dataset_name, config)
        else:
            run_dataset(args.method, dataset_name, config, limit=args.limit)


if __name__ == "__main__":
    main()
