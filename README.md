# Event-level earthquake--blast classification

Analysis code for the manuscript **“Event-Level Earthquake--Blast Discrimination with Variational Waveform Representations and Physical Features: Transfer from Fujian to Utah.”**

The workflow learns three-component waveform representations with an unsupervised variational autoencoder (VAE), aggregates station features to the event level, combines them with physically motivated waveform descriptors, and classifies events using LightGBM.

## Contents

- `src/train_vae.py`: residual three-component VAE and VAE-only LightGBM classifier.
- `src/physical_features.py`: 42 station descriptors and event median/interquartile-range aggregation.
- `src/train_fusion.py`: physical-only and combined VAE-plus-physical LightGBM training.
- `src/evaluate_utah.py`: Utah waveform retrieval, preprocessing, event aggregation, and external evaluation.
- `src/metrics.py`: event-level performance metrics and stratified bootstrap intervals.
- `src/analyze_utah_spatial.py`: paired statistical and spatial-cluster analysis.
- `src/analyze_fujian_paired.py`: paired analysis for the held-out Fujian test set.

This repository intentionally contains no waveform data, event catalog, event inventory, event-level prediction table, trained model, access credential, or local machine path.

## Data access

The controlled Fujian waveform and catalog data are not redistributed. They can be requested through the National Earthquake Science Data Center at <https://data.earthquake.cn>, subject to platform review and applicable data-use conditions.

Utah event labels and picks come from the University of Utah Seismograph Stations discrimination datasets cited in the manuscript. The evaluation script retrieves eligible waveforms from the EarthScope FDSN service after the user supplies the reference file list and catalog.

## Expected Fujian input

The training scripts expect:

```text
dataset/
  manifest.csv
  waveforms.h5
```

`waveforms.h5` must contain a dataset named `waveforms` with shape `[station_records, 3, 4000]` in ENZ order. `manifest.csv` must contain `event_id`, `waveform_index`, `source_type`, `label`, and `split`. All records belonging to one event must share the same split.

The manuscript preprocessing uses a P-5 to P+35 s window sampled at 100 Hz, simple detrending, a second-order zero-phase 1--25 Hz Butterworth bandpass, per-component peak normalization, and a minimum SNR of 3 dB on every component.

## Installation

Python 3.11 was used for the archived analysis.

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

Install the PyTorch build appropriate for the local CUDA driver when GPU execution is required.

## Training

```bash
python src/train_vae.py \
  --dataset PATH/TO/FUJIAN_DATASET \
  --output-dir models/event_scheme_a_v1

python src/train_fusion.py \
  --dataset PATH/TO/FUJIAN_DATASET \
  --vae-model-dir models/event_scheme_a_v1 \
  --output-dir models/event_scheme_a_physics_fusion_v1
```

## Utah evaluation

```bash
python src/evaluate_utah.py \
  --vertical-dir data/Utah_test \
  --catalog data/utah_catalog.csv \
  --baseline-model-dir models/event_scheme_a_v1 \
  --fusion-model-dir models/event_scheme_a_physics_fusion_v1 \
  --output-dir outputs/utah_physics_fusion_v1
```

The evaluation uses the Fujian-selected thresholds. Utah labels are not used for model fitting, feature selection, calibration, or threshold selection.

## Reproducibility boundaries

- Classification and uncertainty calculations use the event as the analysis unit.
- Fujian partitions must be event-disjoint.
- Coordinates, station identifiers, origin times, magnitudes, and catalog descriptions are excluded from classifier inputs.
- Result tables and trained models are generated locally and remain excluded from Git tracking by `.gitignore`.

## License

The code is released under the MIT License. Data remain subject to their original providers' access and reuse conditions.
