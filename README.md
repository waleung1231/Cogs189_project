# COGS189 RSVP EEG Project

This repository contains:
- `run_rsvp.py`: PsychoPy RSVP task for EEG collection with OpenBCI Cyton.
- `train_attention_classifier.py`: feature extraction and classifier training for focused vs zoned-out target trials.

## Data outputs
Session data is saved under:
- `data/rsvp/sub-XX/ses-YY/`

Generated files include:
- `eeg_raw.npy`
- `raw_timestamps.npy`
- `eeg_epochs.npy`
- `events.npy`
- `metadata.csv`
- `session_config.json`

## Run experiment
```bash
python3 run_rsvp.py
```

## Train model (single session)
```bash
python3 train_attention_classifier.py \
  --data_dir data/rsvp/sub-01/ses-01 \
  --out_model cache/attention_model.pkl
```

## Train model (all sessions, group-aware CV)
```bash
python3 train_attention_classifier.py \
  --data_root data/rsvp \
  --out_model cache/attention_model.pkl
```

## Train model (focused-channel subset)
```bash
python3 train_attention_classifier.py \
  --data_root data/rsvp \
  --channels focused_core \
  --out_model cache/attention_model_focused.pkl
```

`--channels` accepts preset names (`all`, `focused_core`, `drop_o1`) or explicit lists like `P3,Pz,P4,Fz`.

## Compare channel sets quickly
```bash
python3 train_attention_classifier.py \
  --data_root data/rsvp \
  --channels focused_core \
  --compare_channels drop_o1 \
  --no_plot --no_tune
```

## Tests
```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```
