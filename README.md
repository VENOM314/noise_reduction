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
distant_speech: ratio 0.40
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

Tune FFT, L1, or least-squares parameters on a smaller clip subset:

```powershell
python experiments/tune_parameters.py --method fft_threshold --dataset all --clean-clip-limit 2
python experiments/tune_parameters.py --method l1_norm --dataset all --clean-clip-limit 1 --profile focused
python experiments/tune_parameters.py --method least_squares --dataset all --clean-clip-limit 2 --profile focused
```

L1 tuning is much slower than FFT because each candidate solves many sparse reconstruction windows.

Outputs are organized by method:

```text
outputs/experiments/<method>/<dataset>/denoised/
outputs/experiments/<method>/<dataset>/metrics/
outputs/experiments/<method>/<dataset>/plots/
outputs/experiments/<method>/parameter_sweep/
```

## Tuned Parameters

FFT thresholding was changed from a single full-clip FFT to overlapping window reconstruction. The tuning pass tried `window_size` values `512`, `1024`, and `2048`; `hop_size` values at one-half and one-quarter of the window; `keep_ratio` values from `0.05` through `1.0`; `f_min = 50`; and dataset-specific `f_max` values. The chosen presets are:

```text
speech: keep_ratio 1.0, window_size 1024, hop_size 256, f_min 50, f_max 8000
instrumental_music: keep_ratio 1.0, window_size 1024, hop_size 256, f_min 50, f_max 4000
piano_notes_chords: keep_ratio 1.0, window_size 2048, hop_size 512, f_min 50, f_max 4000
```

L1 tuning focused on the highest-impact parameters: `lambda_reg`, `n_freqs`, and `window_size`, while keeping `grid_type = linear`, `f_min = 50`, `tol = 0.0001`, and `max_iter = 80`. The search tried `window_size` values `512`, `1024`, and `2048`; `n_freqs` values `80`, `120`, and `200`; and `lambda_reg` values from `0.005` to `0.08`. The chosen presets are:

```text
speech: n_freqs 200, f_max 8000, lambda_reg 0.08, window_size 512, hop_size 256
instrumental_music: n_freqs 200, f_max 5000, lambda_reg 0.08, window_size 512, hop_size 256
piano_notes_chords: n_freqs 200, f_max 5000, lambda_reg 0.08, window_size 512, hop_size 256
```

Least-squares tuning focused on `K`, `n_freqs`, `window_size`, `hop_size`, and `f_max`, while keeping `grid_type = linear` and `f_min = 50`. The focused search tried `window_size` values `1024` and `2048`, `n_freqs` values `200` and `500`, dataset-specific `K` ratios, and `f_max` values up to `7000`. Numerically unstable high-frequency candidates were skipped if the pseudoinverse SVD did not converge. The chosen presets are:

```text
speech: n_freqs 500, f_max 7000, K 500, window_size 1024, hop_size 256
instrumental_music: n_freqs 500, f_max 4000, K 100, window_size 2048, hop_size 512
piano_notes_chords: n_freqs 500, f_max 4000, K 25, window_size 2048, hop_size 512
```

Ranked tuning outputs are written under `outputs/experiments/<method>/parameter_sweep/`.

## Adding Methods

New denoising methods should inherit from `DenoisingMethod`, implement `denoise(audio, sample_rate)`, and register in `noise_cancellation/methods/__init__.py`.

Method-specific parameters belong under `methods.<method_name>` in `configs/experiment_parameters.json`.

## Environment

Install dependencies:

```powershell
pip install -r requirements.txt
```

The scripts expect Python 3.10+, NumPy, SciPy, and SoundFile. VS Code launch configs set `cwd` to the workspace root and `PYTHONPATH` to the workspace root.
