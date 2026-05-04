from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from noise_cancellation.audio import load_mono, peak_normalize, resample_to, save_wav


def prepare_instrumental_music_dataset(source_path: Path, output_dir: Path, sample_rate: int, clip_seconds: float, n_clips: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio, source_rate = load_mono(source_path)
    audio = resample_to(audio, source_rate, sample_rate)

    clip_len = int(sample_rate * clip_seconds)
    needed_len = clip_len * n_clips
    if len(audio) < needed_len:
        audio = np.tile(audio, math.ceil(needed_len / len(audio)))

    max_start = max(1, len(audio) - clip_len)
    paths = []
    for index in range(n_clips):
        start = (index * clip_len * 3) % max_start
        clip = peak_normalize(audio[start : start + clip_len], peak=0.8)
        path = output_dir / f"instrumental_music_{index:03d}.wav"
        save_wav(path, clip, sample_rate)
        paths.append(path)
    return paths
