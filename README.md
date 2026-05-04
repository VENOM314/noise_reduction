# Noise Cancellation

Computational mathematics project for denoising local audio datasets with pluggable methods, then evaluating results with SNR and spectrogram correlation.

## Repository Layout

- `noise_cancellation/methods/`: denoising methods. `base.py` defines the common method interface.
- `noise_cancellation/data_prep/`: local data preparation helpers for clean piano, speech, instrumental music, and noise mixtures.
- `noise_cancellation/audio.py`, `metrics.py`, and `plots.py`: shared audio IO, evaluation metrics, and SVG plotting utilities.
- `experiments/prepare_data.py`: refreshes generated local data from `data/raw/`.
- `experiments/run_experiments.py`: evaluates a selected method against prepared local data.
- `.vscode/launch.json`: VS Code launch configs for data prep, method runs, and parameter sweeps.
- `data/`: tracked local dataset folder.
- `outputs/`: local generated results folder, ignored by Git.

## Methods

Currently registered methods:

```text
least_squares
fft_threshold
l1_norm
```

Each method inherits from `DenoisingMethod` in `noise_cancellation/methods/base.py` and is registered in `noise_cancellation/methods/__init__.py`.

## Data

The experiment runner assumes prepared local data already exists.

```text
data/raw/speech/mini_librispeech/
data/raw/instrumental_music/reverie.ogg
data/raw/noise/subway/
data/clean/<dataset>/
data/noise/<noise_type>/
data/noisy/<dataset>/<noise_type>/
```

Datasets:

```text
piano_notes_chords
speech
instrumental_music
```

Noise types:

```text
gaussian: ratio 0.20
distant_speech: ratio 0.20
subway: ratio 0.60
```

Refresh generated local data only when needed:

```powershell
python experiments/prepare_data.py --dataset all
```

## Running Experiments

Run a method on all datasets:

```powershell
python experiments/run_experiments.py --method least_squares --dataset all
python experiments/run_experiments.py --method fft_threshold --dataset all
python experiments/run_experiments.py --method l1_norm --dataset all
```

Run one dataset:

```powershell
python experiments/run_experiments.py --method least_squares --dataset speech
python experiments/run_experiments.py --method least_squares --dataset instrumental_music
python experiments/run_experiments.py --method least_squares --dataset piano_notes_chords
```

For quicker runs, add `--limit <N>`:

```powershell
python experiments/run_experiments.py --method least_squares --dataset speech --limit 5
```

The VS Code `Run ...` launch configs include commented `--limit` arguments you can uncomment and edit.

Parameter sweep for least squares:

```powershell
python experiments/run_experiments.py --method least_squares --dataset all --parameter-sweep
```

Outputs are organized by method:

```text
outputs/experiments/<method>/<dataset>/denoised/
outputs/experiments/<method>/<dataset>/metrics/
outputs/experiments/<method>/<dataset>/plots/
outputs/experiments/<method>/parameter_sweep/
```

## Adding Methods

New denoising methods should inherit from `DenoisingMethod`, implement `denoise(audio, sample_rate)`, and register in `noise_cancellation/methods/__init__.py`.

Method-specific parameters belong under `methods.<method_name>` in `configs/experiment_parameters.json`.

## Environment

Install dependencies:

```powershell
pip install -r requirements.txt
```

The scripts expect Python 3.10+, NumPy, SciPy, and SoundFile. VS Code launch configs set `cwd` to the workspace root and `PYTHONPATH` to the workspace root.
