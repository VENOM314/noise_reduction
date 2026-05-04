from __future__ import annotations

from pathlib import Path

import numpy as np

from noise_cancellation.audio import peak_normalize, save_wav


def midi_to_freq(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69.0) / 12.0)


def piano_note(freq: float, duration: float, sample_rate: int) -> np.ndarray:
    t = np.arange(int(sample_rate * duration)) / sample_rate
    audio = (
        1.0 * np.sin(2 * np.pi * freq * t)
        + 0.45 * np.sin(2 * np.pi * 2 * freq * t)
        + 0.22 * np.sin(2 * np.pi * 3 * freq * t)
        + 0.10 * np.sin(2 * np.pi * 4 * freq * t)
    )
    attack = max(1, int(sample_rate * 0.02))
    envelope = np.exp(-2.2 * t)
    envelope[:attack] *= np.linspace(0, 1, attack)
    return peak_normalize(audio * envelope, peak=1.0)


def piano_chord_progression(root_midi: int, intervals: list[int], note_duration: float, chord_duration: float, sample_rate: int) -> np.ndarray:
    notes = [root_midi + interval for interval in intervals]
    freqs = [midi_to_freq(note) for note in notes]
    pause = np.zeros(int(sample_rate * 0.25))

    parts = []
    for freq in freqs:
        parts.append(piano_note(freq, note_duration, sample_rate))
        parts.append(pause)

    chord = np.zeros(int(sample_rate * chord_duration))
    for freq in freqs:
        chord += piano_note(freq, chord_duration, sample_rate)
    parts.append(peak_normalize(chord, peak=1.0))

    return peak_normalize(np.concatenate(parts), peak=0.8)


def prepare_piano_note_dataset(output_dir: Path, sample_rate: int, n_clips: int = 20) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    rng = np.random.default_rng(123)
    roots = [48, 50, 52, 53, 55, 57, 59, 60]
    chord_types = {
        "major": [0, 4, 7],
        "minor": [0, 3, 7],
        "dominant7": [0, 4, 7, 10],
        "major7": [0, 4, 7, 11],
        "minor7": [0, 3, 7, 10],
    }
    for index in range(n_clips):
        chord_name = list(chord_types)[index % len(chord_types)]
        root = int(rng.choice(roots))
        audio = piano_chord_progression(
            root,
            chord_types[chord_name],
            note_duration=0.65,
            chord_duration=1.5,
            sample_rate=sample_rate,
        )
        path = output_dir / f"piano_chord_{index:03d}_{chord_name}.wav"
        save_wav(path, audio, sample_rate)
        paths.append(path)
    return paths
