# Data Layout

This folder stores local datasets and generated WAV files. It is tracked by Git so experiments can run from prepared local data.

## Raw Sources

- `raw/speech/mini_librispeech/`: extracted Mini LibriSpeech FLAC speech files.
- `raw/instrumental_music/reverie.ogg`: local instrumental music source.
- `raw/noise/subway/`: extracted DEMAND subway WAV files.

## Generated Data

- `clean/piano_notes_chords/`: 10 generated clean samples. Each file plays notes individually, then the full chord.
- `clean/speech/`: 10 clean 8-second speech samples sliced from local Mini LibriSpeech.
- `clean/instrumental_music/`: 10 clean 8-second instrumental music samples sliced from the local source.
- `noise/gaussian/`: generated Gaussian noise.
- `noise/distant_speech/`: speech babble with filtering and reverb to sound distant.
- `noise/subway/`: subway noise prepared from the local DEMAND source.
- `noisy/<clean_category>/<noise_type>/`: mixed noisy samples and the exact added noise tracks.

Noise ratios are fixed: gaussian and distant speech use quiet ratio `0.20`; subway uses medium ratio `0.60`.

Prepare or refresh generated data with:

```powershell
python experiments/prepare_data.py --dataset all
```
