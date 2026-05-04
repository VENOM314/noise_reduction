from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from noise_cancellation.audio import load_mono, resample_to
from noise_cancellation.config import load_config, output_root
from noise_cancellation.data_prep.datasets import DATASET_NAMES, normalize_dataset_name
from noise_cancellation.methods import create_method
from noise_cancellation.metrics import snr_db, spectrogram_corr

from experiments.run_experiments import collect_noisy_paths, experiment_dirs, parse_noisy_stem, summarize_rows, write_csv


def fft_candidates(dataset_name: str) -> list[dict[str, Any]]:
    f_max_values = {
        "speech": (3000, 4000, 6000, 8000),
        "instrumental_music": (4000, 5000, 8000),
        "piano_notes_chords": (4000, 5000, 8000),
    }
    candidates = []
    for window_size in (512, 1024, 2048):
        for hop_divisor in (2, 4):
            for keep_ratio in (0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.75, 1.0):
                for f_max in f_max_values[dataset_name]:
                    candidates.append(
                        {
                            "keep_ratio": keep_ratio,
                            "window_size": window_size,
                            "hop_size": window_size // hop_divisor,
                            "f_min": 50,
                            "f_max": f_max,
                        }
                    )
    return candidates


def l1_candidates(dataset_name: str, config: dict[str, Any], profile: str) -> list[dict[str, Any]]:
    base = config["methods"]["l1_norm"]["parameter_presets"][dataset_name]
    if profile == "focused":
        window_sizes = (512,)
        n_freqs_values = (80, 120, 200)
        lambda_values = (0.01, 0.02, 0.04, 0.08)
    else:
        window_sizes = (512, 1024, 2048)
        n_freqs_values = (80, 120, 200)
        lambda_values = (0.005, 0.01, 0.02, 0.04)

    candidates = []
    for window_size in window_sizes:
        for n_freqs in n_freqs_values:
            for lambda_reg in lambda_values:
                candidates.append(
                    {
                        "grid_type": "linear",
                        "n_freqs": n_freqs,
                        "f_min": 50,
                        "f_max": base["f_max"],
                        "lambda_reg": lambda_reg,
                        "max_iter": 80,
                        "tol": 0.0001,
                        "window_size": window_size,
                        "hop_size": window_size // 2,
                    }
                )
    return candidates


def method_candidates(method_name: str, dataset_name: str, config: dict[str, Any], profile: str) -> list[dict[str, Any]]:
    if method_name == "fft_threshold":
        return fft_candidates(dataset_name)
    if method_name == "l1_norm":
        return l1_candidates(dataset_name, config, profile)
    raise ValueError("Tuning candidates are defined for fft_threshold and l1_norm.")


def load_eval_items(
    method_name: str,
    dataset_name: str,
    config: dict[str, Any],
    clean_clip_limit: int,
) -> list[dict[str, Any]]:
    dirs = experiment_dirs(method_name, dataset_name)
    sample_rate = int(config["sample_rate"])
    clean_stems = {path.stem for path in sorted(dirs["clean"].glob("*.wav"))[:clean_clip_limit]}
    items = []
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
        length = min(len(clean), len(noisy))
        clean, noisy = clean[:length], noisy[:length]
        items.append(
            {
                "file": clean_stem,
                "noise_type": noise_type,
                "clean": clean,
                "noisy": noisy,
                "snr_noisy": snr_db(clean, noisy),
                "spectrogram_corr_noisy": spectrogram_corr(clean, noisy, sample_rate),
            }
        )
    if not items:
        raise FileNotFoundError(f"No prepared tuning items found for {dataset_name}.")
    return items


def score_candidate(
    method_name: str,
    dataset_name: str,
    config: dict[str, Any],
    params: dict[str, Any],
    items: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_rate = int(config["sample_rate"])
    method = create_method(method_name, config, dataset_name, overrides=params)
    rows = []
    for item in items:
        denoised = method.denoise(item["noisy"], sample_rate)
        length = min(len(item["clean"]), len(item["noisy"]), len(denoised))
        clean = item["clean"][:length]
        denoised = denoised[:length]
        snr_denoised = snr_db(clean, denoised)
        spec_denoised = spectrogram_corr(clean, denoised, sample_rate)
        rows.append(
            {
                "noise_type": item["noise_type"],
                "snr_improvement": snr_denoised - item["snr_noisy"],
                "spectrogram_corr_denoised": spec_denoised,
                "spectrogram_corr_delta": spec_denoised - item["spectrogram_corr_noisy"],
            }
        )

    metric_fields = ["snr_improvement", "spectrogram_corr_denoised", "spectrogram_corr_delta"]
    summary_rows = summarize_rows(rows, ["noise_type"], metric_fields)
    means = {field: mean(row[field] for row in summary_rows) for field in metric_fields}
    ranking_row = {
        "mean_snr_improvement": means["snr_improvement"],
        "mean_spectrogram_corr_denoised": means["spectrogram_corr_denoised"],
        "mean_spectrogram_corr_delta": means["spectrogram_corr_delta"],
    }
    return ranking_row, summary_rows


def tune_method_dataset(
    method_name: str,
    dataset_name: str,
    config: dict[str, Any],
    clean_clip_limit: int,
    profile: str,
) -> None:
    candidates = method_candidates(method_name, dataset_name, config, profile)
    items = load_eval_items(method_name, dataset_name, config, clean_clip_limit)
    print(f"Tuning {method_name} on {dataset_name}: {len(candidates)} {profile} candidates, {len(items)} noisy files")

    ranked_rows = []
    summary_output_rows = []
    start_time = perf_counter()
    for candidate_id, params in enumerate(candidates, start=1):
        ranking_row, summary_rows = score_candidate(method_name, dataset_name, config, params, items)
        ranking_row.update(
            {
                "method": method_name,
                "dataset": dataset_name,
                "candidate_id": candidate_id,
                "params": json.dumps(params, sort_keys=True),
            }
        )
        ranked_rows.append(ranking_row)
        for row in summary_rows:
            summary_output_rows.append(
                {
                    "method": method_name,
                    "dataset": dataset_name,
                    "candidate_id": candidate_id,
                    "params": ranking_row["params"],
                    **row,
                }
            )
        if candidate_id % 10 == 0 or candidate_id == len(candidates):
            elapsed = perf_counter() - start_time
            print(f"  evaluated {candidate_id}/{len(candidates)} candidates in {elapsed:.1f}s")

    ranked_rows.sort(
        key=lambda row: (row["mean_snr_improvement"], row["mean_spectrogram_corr_denoised"]),
        reverse=True,
    )
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank

    sweep_dir = output_root(PROJECT_ROOT) / method_name / "parameter_sweep"
    suffix = "" if profile == "coarse" else f"_{profile}"
    ranked_csv = sweep_dir / f"{dataset_name}_tuning{suffix}_ranked.csv"
    summary_csv = sweep_dir / f"{dataset_name}_tuning{suffix}_by_noise.csv"
    write_csv(
        ranked_csv,
        ranked_rows,
        [
            "rank",
            "method",
            "dataset",
            "candidate_id",
            "mean_snr_improvement",
            "mean_spectrogram_corr_denoised",
            "mean_spectrogram_corr_delta",
            "params",
        ],
    )
    write_csv(
        summary_csv,
        summary_output_rows,
        [
            "method",
            "dataset",
            "candidate_id",
            "params",
            "noise_type",
            "count",
            "snr_improvement",
            "spectrogram_corr_denoised",
            "spectrogram_corr_delta",
        ],
    )

    print(f"Wrote ranked candidates: {ranked_csv}")
    for row in ranked_rows[:5]:
        print(
            f"  rank {row['rank']}: snr +{row['mean_snr_improvement']:.3f}, "
            f"spec {row['mean_spectrogram_corr_denoised']:.3f}, params {row['params']}"
        )


def main() -> None:
    config = load_config(PROJECT_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("fft_threshold", "l1_norm"), required=True)
    parser.add_argument("--dataset", choices=("all", *DATASET_NAMES), default="all")
    parser.add_argument("--clean-clip-limit", type=int, default=2)
    parser.add_argument("--profile", choices=("coarse", "focused"), default="coarse")
    args = parser.parse_args()

    selected_datasets = DATASET_NAMES if args.dataset == "all" else (normalize_dataset_name(args.dataset),)
    for dataset_name in selected_datasets:
        tune_method_dataset(args.method, dataset_name, config, args.clean_clip_limit, args.profile)


if __name__ == "__main__":
    main()
