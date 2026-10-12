# TC-Genesis-Research --- Reproducibility Guide

## 1. Purpose and source basis

This document defines the reproducibility procedure for the tropical
cyclogenesis research software. It is based on the project pipeline
document and the
`Tropical_Cyclogenesis_Project_Complete_Filewise_Explanation (1)(3).docx`
file.

The documented research chain is:

Global GRIB atmospheric data → feature extraction → feature inspection →
IBTrACS labelling → chronological splitting → training-only
normalization → dynamic 21 × 21 × 137 patch generation → CNN-LSTM →
Conv-ViT-BiGRU → gated ensemble → holdout evaluation → prediction
logs/spatial analysis → XAI → runtime configuration → historical FNL
verification → XAI/environment post-classifier → active-system update →
`main.py` → GeoJSON → dashboard.

This guide deliberately records source-document inconsistencies instead
of silently correcting them.

## 2. Required research objects

A full reproduction requires:

### Source data

-   Global atmospheric GRIB analysis archive.
-   IBTrACS v04r01 best-track CSV.
-   Historical NCEP FNL GRIB2 files for verification.
-   Current official cyclone/disturbance information for active-system
    filtering.

### Derived data

-   `data/extracted_features_entire_world/`
-   `tc_24h.csv`, `tc_48h.csv`, `tc_72h.csv`
-   chronological split CSVs
-   `test_predictions_log_24h.csv`
-   historical verification outputs
-   `data/post_classifier_training/genesis_post_classifier_training.csv`
-   `data/active_systems.json`

### Model/configuration artifacts

-   CNN-LSTM checkpoint
-   Conv-ViT-BiGRU checkpoint
-   ensemble checkpoint
-   `channel_means.npy`
-   `channel_stds.npy`
-   `channel_names.json`
-   `training_metadata.json`
-   `operational_thresholds.json`
-   `genesis_post_classifier.json`

### Software

-   repository source
-   `requirements.txt`
-   `Dockerfile`
-   `main.py`
-   `scripts/`
-   `index.html`

## 3. Environment

The documented workflow uses a CUDA-enabled Docker image.

``` bash
docker build -t tc-dl-pipeline:latest .
```

Record the exact: - Python version - TensorFlow/Keras versions -
NumPy/pandas/xarray/netCDF4/SciPy/scikit-learn versions -
Cartopy/cfgrib/ecCodes versions - Docker version - NVIDIA driver/CUDA
runtime - GPU model and memory - CPU/RAM/storage - operating system -
Git commit

The historical pipeline contains institutional proxy credentials in
example commands. **Do not copy those credentials into the repository or
this document.** Supply proxy settings through the runtime environment.

## 4. Input representation

Each model sample is a `21 × 21 × 137` atmospheric tensor.

Seven pressure-level variables are sampled at 19 levels from 1000 to 200
hPa:

``` text
ugrdprs vgrdprs vvelprs tmpprs rhprs hgtprs absvprs
```

This produces 133 pressure-level channels.

Four surface/context channels are:

``` text
pressfc capesfc tmpsfc landmask
```

Total: 137 channels.

The documented geographic grid is latitude `-90° to 90°`, 181 points,
longitude `0° to 359°`, 360 points, with 19 pressure levels.

## 5. Feature extraction and inspection

### Feature extraction

Script:

``` text
scripts/extract_environmental_features_without_labels.py
```

Input:

``` text
data/entire dataset in grib/
```

Output:

``` text
data/extracted_features_entire_world/
```

The helper functions are supplied by the project helper module.

### Feature inspection

Script:

``` text
scripts/check_features.py
```

Verify: - complete global coverage; - latitude/longitude ordering; - all
11 required variables; - all 19 pressure levels; - expected dimensions.

Do not proceed if the schema is wrong.

## 6. IBTrACS labelling

Source:

``` text
data/ibtracs/ibtracs.ALL.list.v04r01.csv
```

Scripts:

``` text
scripts/create_labels.py
scripts/create_labels1.py
```

The documented label construction preserves: - all active cyclone
locations; - active cyclone IDs; - simultaneous genesis events; -
JSON-safe coordinate lists; - 0°--360° longitude; - lead-time-specific
labels; - non-genesis atmospheric states.

Outputs:

``` text
data/extracted_features_entire_world/tc_24h.csv
data/extracted_features_entire_world/tc_48h.csv
data/extracted_features_entire_world/tc_72h.csv
```

For the principal experiment, use `tc_24h.csv`.

## 7. Chronological split

Script:

``` text
scripts/split_data.py
```

Split by year rather than random sampling.

Expected artifacts:

``` text
tc_24h_train.csv
tc_24h_val.csv
tc_24h_test.csv
```

**Important source discrepancy:** the filewise explanation contains one
statement saying training is `2000–2018`, while the current
research/manuscript description uses `2008–2018`, with validation
`2019–2020` and test `2021–2022`. Before final archival release, inspect
the actual split CSVs and record the exact years used for the published
result. Do not silently change the experiment.

## 8. Normalization

Script:

``` text
scripts/precompute_stats.py
```

Compute channel statistics from training data only.

Artifacts:

``` text
saved_models/channel_means.npy
saved_models/channel_stds.npy
```

The filewise explanation describes 136 meteorological channels being
normalized while the land-mask channel remains unchanged; the model
input still contains 137 total channels.

These exact arrays must be archived with the model release.

## 9. Dynamic generator

Script:

``` text
scripts/tcdl_generator.py
```

`TropicalCyclogenesisGenerator`: - reads split CSVs; - loads atmospheric
feature files; - extracts `21 × 21 × 137` patches; - creates difficult
negative samples; - applies safe training augmentation; - normalizes
using fixed statistics; - streams batches without loading the complete
dataset into memory.

Validation and test generation must be deterministic and must not use
random training augmentation.

## 10. CNN-LSTM

Scripts:

``` text
scripts/train_cnn_lstm.py
scripts/cnn_lstm.py
```

The first expert uses convolutional feature extraction followed by
row-wise and column-wise recurrent processing.

For reproducibility, archive: - exact source commit; - model
checkpoint; - random seed(s); - batch size; - epochs; - optimizer; -
learning rate; - loss/focal-loss settings; - gradient clipping; -
checkpoint/early-stopping settings; - generator configuration.

## 11. Conv-ViT-BiGRU

Script:

``` text
scripts/train_vit_gru.py
```

The documented model combines: - convolutional stem; -
squeeze-and-excitation channel attention; - patch/token
representation; - positional embedding; - Transformer blocks; -
multi-head attention; - BiGRU; - dense classification head.

The pipeline document gives a representative 24-hour command using
`batch-size 36` and `workers 8`. Archive the exact command used for the
final published checkpoint.

## 12. Ensemble

Script:

``` text
scripts/train_ensemble.py
```

The ensemble loads both experts and learns gated cross-attention fusion.
It is not a simple average.

Archive: - both expert checkpoints; - ensemble checkpoint; - exact
architecture revision; - seeds; - optimizer/loss; - learning rates; -
freezing/fine-tuning stages; - batch size; - checkpoint selected for
evaluation.

## 13. Holdout evaluation

Scripts:

``` text
scripts/evaluate_model.py
scripts/evaluate_ensemble.py
```

Evaluate against the untouched chronological test set.

Record: - accuracy - precision - recall - F1 - PR-AUC - ROC-AUC -
confusion matrix

The same test split must be used for both experts and the ensemble.

Do not tune the model using the test set.

## 14. Test prediction log and spatial analysis

Script:

``` text
scripts/log_test_predictions.py
```

Output:

``` text
data/extracted_features_entire_world/test_predictions_log_24h.csv
```

The log preserves actual labels and predicted probabilities for error
inspection.

Script:

``` text
scripts/plot_spatial_performance.py
```

Use the log to map hits, misses, and false alarms across tropical
basins.

## 15. XAI

The documented research XAI workflow includes: - Grad-CAM; - Integrated
Gradients; - Vision Transformer attention rollout; - permutation
importance; - vertical/pressure-level attribution.

The operational post-classifier uses mean absolute input-gradient
features.

For every published XAI result, record: - checkpoint; - preprocessing; -
method; - selected samples; - attribution settings; - aggregation
procedure; - output figure/file.

Do not substitute a different checkpoint for a published XAI figure
without recording the change.

## 16. Runtime configuration

Scripts:

``` text
scripts/generate_validation_and_configs.py
scripts/generate_runtime_configs.py
```

The documented validation procedure selects an operational threshold
from validation predictions.

Representative documented configuration:

``` text
leadtime: 24
selection: min_recall
target_recall: 0.90
batch-size: 32
```

Outputs:

``` text
channel_names.json
training_metadata.json
operational_thresholds.json
```

`channel_names.json` is critical because it fixes the exact 137-channel
order.

`training_metadata.json` records training-domain information used for
domain-shift safeguards.

`operational_thresholds.json` records validation-selected detection
thresholds.

## 17. Historical FNL verification

Script:

``` text
scripts/run_fnl_2026_cyclogenesis_verification.py
```

The script can operate on supplied historical date ranges despite its
filename.

Workflow:

1.  obtain/reuse native FNL cycles;
2.  run global ensemble inference;
3.  cluster detections;
4.  match against IBTrACS;
5.  record hits, misses, false alarms;
6.  calculate precision, POD, FAR and CSI.

The documented example includes:

``` text
leadtime 24
alert threshold 0.75
minimum expert confidence 0.45
minimum mid-level RH 50%
minimum 850-hPa vorticity 7.5e-6 s^-1
maximum pressure anomaly 0 Pa
maximum 200–850-hPa shear 25 m/s
minimum cluster support 2
cluster radius 700 km
maximum detections 2
IBTrACS distance tolerance 800 km
batch size 256
```

These are verification settings and should not be confused with every
operational setting.

## 18. FNL cache

Directory:

``` text
data/fnl_cache/
```

It stores downloaded FNL GRIB2 cycles for reuse.

For archival reproducibility, record: - dates/cycles; - source; -
download manifest; - checksums where possible.

Do not place large raw archives in GitHub unless explicitly intended.

## 19. XAI/environment post-classifier

Script:

``` text
scripts/build_post_classifier_fast.py
```

The documented process: 1. collect historical predictions; 2. identify
HIT and FALSE_ALARM detections; 3. calculate environmental diagnostics;
4. calculate batched XAI features; 5. train logistic regression; 6.
compare against the deterministic fallback; 7. retain the learned model
only when it satisfies the documented selection rule.

Documented example:

``` text
target samples 100
minimum hits 20
minimum false alarms 40
start year 2025
start month 7
months 5
batch size 256
XAI batch size 16
```

Completed training set:

``` text
110 detections
31 hits
79 false alarms
```

Outputs:

``` text
data/post_classifier_training/genesis_post_classifier_training.csv
saved_models/genesis_post_classifier.json
```

The JSON stores feature names, coefficients, intercept, means, scales,
imputation medians, positive threshold and high-priority threshold.

## 20. Active-system update

Script:

``` text
scripts/update_active_systems.py
```

Output:

``` text
data/active_systems.json
```

Update this before operational inference so known active cyclones are
not reported as new genesis.

## 21. Operational inference

Script:

``` text
main.py
```

Inputs include: - latest FNL; - CNN-LSTM; - Conv-ViT-BiGRU; -
ensemble; - normalization; - channel schema; - training metadata; -
operational thresholds; - post-classifier; - active systems.

The pipeline performs: 1. FNL acquisition; 2. schema/metadata
validation; 3. preprocessing; 4. global sliding-window inference; 5.
candidate filtering/NMS; 6. environmental diagnostics; 7. XAI features;
8. active-system matching; 9. learned post-classification or fallback;
10. GeoJSON export.

Representative documented 24-hour command:

``` bash
python main.py \
  --once \
  --lead-times 24 \
  --alert-threshold 0.75 \
  --min-expert-confidence 0.45 \
  --nms-radius-km 500 \
  --max-detections 5 \
  --classification-mode auto
```

The historical pipeline also shows `--batch-size 256`,
`--post-classifier-model`, `--active-systems-json`, and `--insecure-ssl`
in the full operational command. Record the exact command used in the
final release.

## 22. Domain shift

Historical model training uses the documented FNL/NCEP atmospheric
representation. Operational ingestion can involve GDAS/GFS fields.

This can create training-to-inference domain shift.

The operational implementation contains a domain-shift safeguard. Do not
bypass it without validation or bias correction.

## 23. GeoJSON and dashboard

GeoJSON:

``` text
output/latest_24h.geojson
```

and dated outputs.

Records may include: - latitude/longitude; - raw neural score; - expert
scores; - environmental diagnostics; - XAI values; - final class; -
classification reason; - learned probability; - classification source.

Dashboard:

``` text
index.html
```

The dashboard loads the GeoJSON and provides interactive map-based
review.

## 24. Reproducibility checks

Before release, verify:

-   [ ] exact source-data versions recorded;
-   [ ] IBTrACS v04r01 recorded;
-   [ ] FNL source and verification period recorded;
-   [ ] all 11 variables present;
-   [ ] all 19 pressure levels present;
-   [ ] 137-channel order matches `channel_names.json`;
-   [ ] exact train/validation/test years recorded;
-   [ ] no test data used for tuning;
-   [ ] normalization derived only from training data;
-   [ ] validation/test augmentation disabled;
-   [ ] exact CNN-LSTM checkpoint preserved;
-   [ ] exact Conv-ViT-BiGRU checkpoint preserved;
-   [ ] exact ensemble checkpoint preserved;
-   [ ] random seeds recorded;
-   [ ] software versions recorded;
-   [ ] test prediction log archived;
-   [ ] XAI checkpoint/settings archived;
-   [ ] threshold configuration archived;
-   [ ] post-classifier JSON archived;
-   [ ] active-system input recorded for operational runs;
-   [ ] domain-shift safeguard retained.

## 25. Recommended archival package

Archive together:

``` text
Source-code Git commit/tag
README.md
REPRODUCIBILITY.md
CITATION.cff
requirements.txt
Dockerfile

CNN-LSTM checkpoint
Conv-ViT-BiGRU checkpoint
Ensemble checkpoint

channel_means.npy
channel_stds.npy
channel_names.json
training_metadata.json
operational_thresholds.json
genesis_post_classifier.json

Final train/validation/test split definitions
Test prediction log
Post-classifier training CSV
Representative XAI figures
Evaluation figures
Example GeoJSON
Dataset access/download manifest
```

Large source datasets should normally remain with their authoritative
data repositories and be referenced by version/DOI rather than copied
into GitHub.

## 26. Provenance record

Every release should record:

``` text
Repository:
Git commit:
Git tag/release:
Docker image:
Python:
TensorFlow/Keras:
CUDA/NVIDIA runtime:
GPU:

Atmospheric dataset:
IBTrACS version:
FNL verification period:

Training years:
Validation years:
Test years:

Normalization artifact:
Channel schema:
Training metadata:
Threshold configuration:

CNN-LSTM checkpoint:
Conv-ViT-BiGRU checkpoint:
Ensemble checkpoint:
Post-classifier JSON:

Test prediction log:
XAI configuration:
Verification configuration:
```

## 27. Known documentation discrepancy

The filewise explanation contains a `2000–2018` training-year statement,
while the current manuscript/project result description uses
`2008–2018`, validation `2019–2020`, and test `2021–2022`.

This must be checked against the actual final split CSVs before
publication/archive.

The release should contain exactly one verified split definition. Do not
regenerate the paper's result merely to resolve a documentation
inconsistency.

## 28. Reproduction levels

### Level 1 --- Software reproduction

Install the repository and execute the documented scripts.

### Level 2 --- Evaluation reproduction

Use archived checkpoints, split definitions, normalization and test
artifacts to reproduce the reported holdout metrics.

### Level 3 --- Training reproduction

Start from archived source data and retrain the experts and ensemble.

### Level 4 --- End-to-end operational reproduction

Reproduce FNL acquisition → preprocessing → inference → filtering/NMS →
XAI/environment features → active-system matching → post-classification
→ GeoJSON → dashboard.

Level 4 is inherently time-dependent because atmospheric analyses and
active-system information change.

## 29. Reproducibility interpretation

Exact numerical identity is expected for deterministic evaluation when
the same inputs, split, preprocessing, normalization, checkpoint,
threshold and code revision are used.

Training from scratch can differ unless random seeds, software versions,
hardware, deterministic settings and data ordering are controlled.

Operational outputs will change as new atmospheric analyses and
active-system data become available.

## 30. Final reproducibility chain

``` text
Global atmospheric GRIB
        ↓
Feature extraction
        ↓
Feature validation
        ↓
IBTrACS genesis labels
        ↓
Chronological split
        ↓
Training-only normalization
        ↓
21 × 21 × 137 patches
        ↓
CNN-LSTM + Conv-ViT-BiGRU
        ↓
Gated ensemble
        ↓
Chronological holdout evaluation
        ↓
Prediction logs + spatial analysis
        ↓
XAI
        ↓
Validation-selected configuration
        ↓
Historical FNL verification
        ↓
XAI/environment post-classifier
        ↓
Active-system update
        ↓
Operational main.py
        ↓
GeoJSON
        ↓
Interactive dashboard
```

The strongest reproducibility record is obtained when the exact code
revision, dataset versions, split definitions, normalization arrays,
model checkpoints, thresholds, XAI configuration and verification
configuration are archived together.
