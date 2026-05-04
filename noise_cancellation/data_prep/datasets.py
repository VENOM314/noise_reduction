from __future__ import annotations

import math
import shutil
import stat
from pathlib import Path
from typing import Any

import numpy as np

from noise_cancellation.audio import (
    load_mono,
    peak_normalize,
    resample_to,
    save_wav,
)

from .audio_transforms import apply_distant_hall_effect, prevent_clipping, scale_noise_to_ratio, slice_or_tile
from .music import prepare_instrumental_music_dataset
from .piano import prepare_piano_note_dataset


DATASET_NAMES = ("piano_notes_chords", "speech", "instrumental_music")
EXPERIMENT_ALIASES = {
    "synthetic_piano_notes": "piano_notes_chords",
    "speech_clean": "speech",
}


def reset_generated_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, onerror=retry_readonly_remove)
    path.mkdir(parents=True, exist_ok=True)


def retry_readonly_remove(function, path: str, _exc_info) -> None:
    Path(path).chmod(stat.S_IWRITE)
    function(path)


def dataset_clip_seconds(dataset_name: str, config: dict[str, Any]) -> float:
    durations = config.get("clip_seconds_by_dataset", {})
    return float(durations.get(dataset_name, 4.0))


def max_clean_clip_seconds(config: dict[str, Any]) -> float:
    durations = config.get("clip_seconds_by_dataset", {})
    if not durations:
        return 4.0
    return float(max(durations.values()))


def data_paths(project_root: Path) -> dict[str, Path]:
    data_root = project_root / "data"
    raw_root = data_root / "raw"
    return {
        "data": data_root,
        "raw": raw_root,
        "clean": data_root / "clean",
        "noise": data_root / "noise",
        "noisy": data_root / "noisy",
        "speech_raw": raw_root / "speech" / "mini_librispeech",
        "music_source": raw_root / "instrumental_music" / "reverie.ogg",
        "subway_raw": raw_root / "noise" / "subway",
    }


def normalize_dataset_name(name: str) -> str:
    return EXPERIMENT_ALIASES.get(name, name)


def require_local_sources(project_root: Path, config: dict[str, Any]) -> None:
    paths = data_paths(project_root)
    missing = []
    if not sorted(paths["speech_raw"].rglob("*.flac")):
        missing.append("Mini LibriSpeech FLAC files under data/raw/speech/mini_librispeech/")
    if not paths["music_source"].exists():
        missing.append("instrumental music source at data/raw/instrumental_music/reverie.ogg")
    if "subway" in config["active_noise_types"] and not sorted(paths["subway_raw"].rglob("*.wav")):
        missing.append("DEMAND subway WAV files under data/raw/noise/subway/")

    if missing:
        details = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(f"Local raw data is missing:\n{details}\nSee README.md for download commands.")


def find_subway_channel(extracted_dir: Path) -> Path:
    wavs = sorted(extracted_dir.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No subway WAV files found in {extracted_dir}")
    for path in wavs:
        lowered = path.stem.lower()
        if "ch01" in lowered or "ch1" in lowered:
            return path
    return wavs[0]


def prepare_clean_speech(speech_raw_root: Path, clean_dir: Path, sample_rate: int, clip_seconds: float, n_clips: int) -> list[Path]:
    reset_generated_dir(clean_dir)
    target_len = int(sample_rate * clip_seconds)
    saved = []
    flac_files = sorted(speech_raw_root.rglob("*.flac"))
    for source_path in flac_files:
        audio, source_rate = load_mono(source_path)
        audio = resample_to(audio, source_rate, sample_rate)
        if len(audio) < target_len:
            continue
        path = clean_dir / f"speech_{len(saved):03d}.wav"
        save_wav(path, peak_normalize(audio[:target_len], peak=0.8), sample_rate)
        saved.append(path)
        if len(saved) >= n_clips:
            break
    if len(saved) < n_clips:
        raise RuntimeError(f"Only prepared {len(saved)} clean speech clips")
    return saved


def prepare_noise_sources(project_root: Path, noise_dir: Path, config: dict[str, Any]) -> dict[str, Path]:
    paths = data_paths(project_root)
    reset_generated_dir(noise_dir)
    sample_rate = int(config["sample_rate"])
    clip_seconds = max_clean_clip_seconds(config)
    n_clips = int(config["n_clean_clips"])
    target_len = int(sample_rate * clip_seconds * n_clips)
    active = set(config["active_noise_types"])
    sources: dict[str, Path] = {}

    if "gaussian" in active:
        path = noise_dir / "gaussian" / "gaussian.wav"
        rng = np.random.default_rng(2026)
        audio = rng.normal(0.0, 1.0, target_len)
        save_wav(path, peak_normalize(audio, peak=0.8), sample_rate)
        sources["gaussian"] = path

    if "distant_speech" in active:
        path = noise_dir / "distant_speech" / "distant_speech.wav"
        flac_files = sorted(paths["speech_raw"].rglob("*.flac"))
        candidate_files = flac_files[n_clips:] or flac_files
        tracks = []
        for source_path in candidate_files:
            audio, source_rate = load_mono(source_path)
            audio = resample_to(audio, source_rate, sample_rate)
            if len(audio) < sample_rate:
                continue
            if len(audio) < target_len:
                audio = np.tile(audio, math.ceil(target_len / len(audio)))
            tracks.append(audio[:target_len])
            if len(tracks) >= 5:
                break
        if len(tracks) < 2:
            raise RuntimeError("Not enough local speech files to build distant speech noise")
        babble = np.mean(np.vstack(tracks), axis=0)
        save_wav(path, apply_distant_hall_effect(babble, sample_rate), sample_rate)
        sources["distant_speech"] = path

    if "subway" in active:
        path = noise_dir / "subway" / "subway.wav"
        audio, source_rate = load_mono(find_subway_channel(paths["subway_raw"]))
        audio = resample_to(audio, source_rate, sample_rate)
        if len(audio) < target_len:
            audio = np.tile(audio, math.ceil(target_len / len(audio)))
        save_wav(path, peak_normalize(audio[:target_len], peak=0.8), sample_rate)
        sources["subway"] = path

    return sources


def mix_noisy_dataset(clean_files: list[Path], noise_sources: dict[str, Path], noisy_root: Path, config: dict[str, Any]) -> list[Path]:
    reset_generated_dir(noisy_root)
    sample_rate = int(config["sample_rate"])
    noise_ratios = config["noise_ratios"]
    noisy_paths = []

    for noise_type, noise_path in noise_sources.items():
        noise_audio, noise_rate = load_mono(noise_path)
        noise_audio = resample_to(noise_audio, noise_rate, sample_rate)
        ratio = float(noise_ratios[noise_type])
        noisy_dir = noisy_root / noise_type / "noisy"
        added_dir = noisy_root / noise_type / "noise_added"
        noisy_dir.mkdir(parents=True, exist_ok=True)
        added_dir.mkdir(parents=True, exist_ok=True)

        for clip_index, clean_path in enumerate(clean_files):
            clean, clean_rate = load_mono(clean_path)
            clean = resample_to(clean, clean_rate, sample_rate)
            max_start = max(1, len(noise_audio) - len(clean))
            offset = (clip_index * len(clean) + len(clean) // 3) % max_start
            noise = slice_or_tile(noise_audio, offset, len(clean))
            scaled_noise = scale_noise_to_ratio(clean, noise, ratio)
            noisy, scaled_noise = prevent_clipping(clean + scaled_noise, scaled_noise)

            stem = f"{clean_path.stem}_{noise_type}"
            noisy_path = noisy_dir / f"{stem}_noisy.wav"
            save_wav(noisy_path, noisy, sample_rate)
            save_wav(added_dir / f"{stem}_noise.wav", scaled_noise, sample_rate)
            noisy_paths.append(noisy_path)
    return sorted(noisy_paths)


def prepare_clean_dataset(project_root: Path, dataset_name: str, clean_dir: Path, config: dict[str, Any]) -> list[Path]:
    paths = data_paths(project_root)
    sample_rate = int(config["sample_rate"])
    n_clips = int(config["n_clean_clips"])
    if dataset_name == "piano_notes_chords":
        reset_generated_dir(clean_dir)
        return prepare_piano_note_dataset(clean_dir, sample_rate, n_clips)
    if dataset_name == "speech":
        return prepare_clean_speech(paths["speech_raw"], clean_dir, sample_rate, dataset_clip_seconds(dataset_name, config), n_clips)
    if dataset_name == "instrumental_music":
        reset_generated_dir(clean_dir)
        return prepare_instrumental_music_dataset(
            paths["music_source"],
            clean_dir,
            sample_rate,
            dataset_clip_seconds(dataset_name, config),
            n_clips,
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def prepare_dataset(
    project_root: Path,
    dataset_name: str,
    config: dict[str, Any],
    noise_sources: dict[str, Path] | None = None,
) -> tuple[list[Path], list[Path], dict[str, Path]]:
    paths = data_paths(project_root)
    require_local_sources(project_root, config)
    clean_dir = paths["clean"] / dataset_name
    noisy_dir = paths["noisy"] / dataset_name
    clean_files = prepare_clean_dataset(project_root, dataset_name, clean_dir, config)
    if noise_sources is None:
        noise_sources = prepare_noise_sources(project_root, paths["noise"], config)
    noisy_paths = mix_noisy_dataset(clean_files, noise_sources, noisy_dir, config)
    return clean_files, noisy_paths, noise_sources


def prepare_all_data(project_root: Path, config: dict[str, Any], dataset_name: str = "all") -> None:
    paths = data_paths(project_root)
    require_local_sources(project_root, config)
    selected = DATASET_NAMES if dataset_name == "all" else (normalize_dataset_name(dataset_name),)
    noise_sources = prepare_noise_sources(project_root, paths["noise"], config)
    for name in selected:
        clean_files, noisy_paths, _ = prepare_dataset(project_root, name, config, noise_sources=noise_sources)
        print(f"Prepared {len(clean_files)} clean and {len(noisy_paths)} noisy files for {name}")
