# Experiments

This folder contains thin command-line entry points. Shared logic lives under top-level `noise_cancellation/`.

## Prepare Data

Refresh generated local data from existing `data/raw/` sources:

```powershell
python experiments/prepare_data.py --dataset all
```

Prepare one dataset:

```powershell
python experiments/prepare_data.py --dataset speech
```

## Run Methods

Evaluate a method on prepared local data:

```powershell
python experiments/run_experiments.py --method least_squares --dataset all
python experiments/run_experiments.py --method fft_threshold --dataset all
python experiments/run_experiments.py --method l1_norm --dataset all
```

Run a single dataset:

```powershell
python experiments/run_experiments.py --method least_squares --dataset instrumental_music
```

Limit the number of noisy files when you want a quicker run:

```powershell
python experiments/run_experiments.py --method least_squares --dataset piano_notes_chords --limit 5
```

Run a least-squares parameter sweep:

```powershell
python experiments/run_experiments.py --method least_squares --dataset all --parameter-sweep
```

Outputs are written to `outputs/experiments/<method>/`.
