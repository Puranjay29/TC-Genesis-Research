# TC-Genesis-Research

## An Explainable Dual-Expert Deep Learning Framework for Global 24-h Tropical Cyclogenesis Prediction

Research software for global tropical cyclogenesis screening using multivariable atmospheric analyses, complementary deep-learning experts, learned feature fusion, Explainable AI (XAI), historical verification, candidate post-classification, and operational GeoJSON generation.

Repository: https://github.com/Puranjay29/TC-Genesis-Research

---

## 1. Overview

Tropical cyclogenesis is the early formation of a tropical cyclone. This project examines global atmospheric conditions and identifies ocean regions that may develop into a tropical cyclone during the next 24 hours.

The complete research and operational workflow is:

1. Global atmospheric GRIB data
2. Environmental feature extraction
3. Feature inspection
4. IBTrACS labelling
5. Chronological train/validation/test splitting
6. Fixed training-set normalization
7. Dynamic patch generation
8. CNN-LSTM training
9. Conv-ViT-BiGRU training
10. Learned ensemble training
11. Holdout evaluation
12. Prediction logging and spatial analysis
13. XAI analysis
14. Runtime configuration
15. Historical FNL verification
16. XAI/environment post-classifier training
17. Active-system updating
18. Operational inference with `main.py`
19. GeoJSON generation and dashboard visualization

The project documentation describes this as a complete pipeline from global weather data and labels through deep-learning models, XAI, verification, post-classification, and dashboard output.

---

## 2. Scientific Input Representation

Each model sample is a **21 × 21 × 137** atmospheric tensor.

### Pressure-level channels

Seven variables are represented at 19 pressure levels from 1000 to 200 hPa:

- `ugrdprs`
- `vgrdprs`
- `vvelprs`
- `tmpprs`
- `rhprs`
- `hgtprs`
- `absvprs`

Thus:

`7 variables × 19 pressure levels = 133 channels`

### Surface/context channels

Four additional fields are included:

- `pressfc`
- `capesfc`
- `tmpsfc`
- `landmask`

Total:

`133 + 4 = 137 channels`

The geographic representation uses a global approximately 1° grid and the project’s 0°–360° longitude convention.

---

## 3. Data Sources

### NCEP FNL

NCEP Final Operational Global Analysis fields provide the atmospheric input and historical verification data.

The complete FNL archive is not redistributed in this repository. Obtain the required data from the official NCAR Research Data Archive and retain the appropriate dataset citation.

### IBTrACS

IBTrACS v04r01 provides observed tropical-cyclone tracks, positions, timestamps, storm IDs, statuses, and genesis information.

The complete IBTrACS archive is not redistributed in this repository. Obtain it from the official NOAA/NCEI source and cite the version used.

---

## 4. Dataset Construction

### Genesis labels

Atmospheric analysis times are matched with IBTrACS to construct lead-time-specific genesis labels.

The label construction preserves:

- all active cyclone locations;
- active cyclone IDs;
- multiple simultaneous genesis events;
- 0°–360° longitude coordinates;
- lead-time information;
- active-system exclusion information.

The workflow supports 24 h, 48 h, and 72 h label products. The principal reported research experiment is the 24-hour classifier.

### Chronological split

The principal 24-hour experiment uses:

| Split | Years | Purpose |
|---|---|---|
| Training | 2008–2018 | Model development |
| Validation | 2019–2020 | Validation/configuration |
| Test | 2021–2022 | Independent chronological holdout |

Year-based splitting is used instead of random splitting to prevent future-season information from leaking into model development.

### Negative sampling

The dynamic generator creates difficult negative samples as well as background negatives. A storm-exclusion radius is used to reduce contamination from already active systems.

### Normalization

Channel-wise means and standard deviations are computed from training data and reused for validation, testing, verification, and operational inference. The land-mask channel remains unchanged.

---

## 5. Model Architecture

### 5.1 Dual-Axis Spatial CNN-LSTM

The first expert combines convolutional feature extraction with row-wise and column-wise recurrent processing.

```text
21 × 21 × 137
      |
Conv2D 64 → BN → ReLU → MaxPool
      |
Conv2D 128 → BN → ReLU → MaxPool
      |
   5 × 5 × 128
      |
 ┌────┴────┐
Row LSTM  Column LSTM
 └────┬────┘
      |
 Concatenate
      |
Dense 128 → Dense 64
      |
   Sigmoid
```

Its purpose is to capture compact local cyclone-like structures and ordered spatial patterns.

Training configuration includes focal binary cross-entropy, Adam, learning rate `1e-4`, gradient clipping, and PR-AUC monitoring.

### 5.2 Conv-ViT-BiGRU

The second expert combines local convolution, squeeze-and-excitation channel attention, Vision Transformer processing, and bidirectional recurrent processing.

```text
21 × 21 × 137
      |
CNN stem 64 → 128
      |
3 × 3 patching → 49 tokens
      |
192-dimensional embeddings
      |
Positional embedding
      |
4 Transformer blocks / 6 heads
      |
BiGRU
      |
Dense 128 → Dense 64
      |
   Sigmoid
```

This expert is designed to represent wider spatial relationships and environmental context.

### 5.3 Learned Ensemble

The ensemble is **not** a simple average.

It:

1. extracts representations from both experts;
2. incorporates expert probabilities;
3. projects expert representations into a common latent space;
4. constructs latent tokens;
5. applies joint self-attention;
6. uses product and absolute-difference features;
7. combines probability-level features;
8. produces the final ensemble probability.

The ensemble is trained in stages, beginning with frozen experts and followed by limited upper-layer fine-tuning.

---

## 6. Explainable AI

The XAI workflow includes:

- Grad-CAM;
- Integrated Gradients;
- Vision Transformer attention rollout;
- permutation sensitivity;
- variable-group attribution;
- pressure-level attribution.

The analysis identifies which atmospheric variables, locations, and pressure levels influence predictions.

Important variables identified in the project analysis include absolute vorticity, relative humidity, vertical velocity, and wind structure.

The operational post-classifier uses mean absolute input-gradient features, while the wider research XAI suite supports multiple attribution methods.

---

## 7. Historical Verification

The historical FNL verification workflow:

1. obtains or reuses FNL analysis cycles before observed genesis;
2. runs the trained ensemble;
3. clusters and filters detections;
4. matches detections against IBTrACS;
5. records hits, misses, and false alarms;
6. calculates verification statistics.

The verification workflow supports precision, probability of detection (POD), false alarm ratio (FAR), and critical success index (CSI).

---

## 8. XAI/Environment Post-Classifier

A second-stage logistic classifier can combine:

- CNN probability;
- ViT probability;
- ensemble probability;
- environmental diagnostics;
- XAI features;
- cyclone-match information.

The documented training set contained:

- 110 labelled detections;
- 31 hits;
- 79 false alarms.

The classifier estimates whether a retained hotspot resembles a historical HIT or FALSE_ALARM. It is a candidate-triage stage and does not replace the neural detector or scientific review.

A deterministic fallback is retained when the learned classifier cannot safely be used.

---

## 9. Operational Inference

`main.py` implements the operational pipeline:

```text
Latest atmospheric analysis
          |
          v
137-channel validation
          |
          v
21 × 21 patch construction
          |
          v
CNN-LSTM ─────────┐
                  +--> Learned ensemble
Conv-ViT-BiGRU ───┘
          |
          v
Candidate filtering + NMS
          |
          v
Environmental diagnostics + XAI
          |
          v
Active-system matching
          |
          v
Learned post-classifier / fallback
          |
          v
GeoJSON
          |
          v
Dashboard
```

The current documented 24-hour operational mode uses:

- detection threshold: `0.75`
- minimum expert confidence: `0.45`
- NMS radius: `500 km`
- maximum detections: `5`
- classification mode: `auto`

### Domain-shift safeguard

Historical training uses NCEP/NCAR-family atmospheric data, while operational ingestion can use GDAS/GFS fields. This creates a potential training-to-inference domain shift.

The operational implementation therefore contains an explicit domain-shift safeguard. Do not bypass it without appropriate overlapping-data validation or bias correction.

---

## 10. Outputs

Operational GeoJSON records can contain:

- latitude and longitude;
- neural/ensemble scores;
- classification confidence;
- classification label;
- classification reason;
- active-system match;
- environmental diagnostics;
- XAI information;
- learned post-classifier probability;
- classification source.

The cleaned repository retains an example forecast:

```text
output/example_forecast/latest_24h.geojson
```

Large runtime output collections are excluded from version control.

---

## 11. Repository Structure

```text
TC-Genesis-Research/
├── main.py
├── Dockerfile
├── requirements.txt
├── README.md
├── CITATION.cff
├── scripts/
│   ├── data preparation and label construction
│   ├── normalization and generators
│   ├── CNN-LSTM training
│   ├── Conv-ViT-BiGRU training
│   ├── ensemble training
│   ├── model evaluation
│   ├── prediction logging
│   ├── spatial analysis
│   ├── XAI analysis
│   ├── runtime configuration
│   ├── historical FNL verification
│   ├── post-classifier construction
│   └── active-system updating
├── saved_models/
├── plots/
├── predictions on test set/
├── output/
│   └── example_forecast/
├── analyze variable  importance/
├── gdex/
└── .gitignore
```

Raw atmospheric data, large derived datasets, FNL caches, logs, secrets, and transient runtime outputs are intentionally excluded.

---

## 12. Installation

### Python

Main dependencies include:

- TensorFlow 2.16.x
- Keras
- NumPy
- pandas
- xarray
- netCDF4
- SciPy
- scikit-learn
- scikit-image
- Matplotlib
- Cartopy
- cfgrib

Install with:

```bash
pip install -r requirements.txt
```

GRIB processing also requires suitable ecCodes libraries.

### Docker

Build the supplied CUDA-enabled image:

```bash
docker build -t tc-dl-pipeline:latest .
```

GPU training/inference requires a compatible NVIDIA GPU and container runtime.

---

## 13. Reproducible Workflow

Recommended execution order:

### 1. Feature extraction

Use the feature-extraction scripts in `scripts/` to convert the source GRIB archive into reusable global feature files.

### 2. Feature inspection

Verify global geographic coverage, variables, pressure levels, dimensions, and longitude ordering.

### 3. IBTrACS labelling

Generate lead-time-specific labelled CSV files and preserve active-system information.

### 4. Chronological splitting

Create train, validation, and test CSV files using year-based separation.

### 5. Normalization

Compute channel statistics from training data only.

### 6. Expert training

Train:

```text
CNN-LSTM
Conv-ViT-BiGRU
```

### 7. Ensemble training

Train the gated fusion ensemble.

### 8. Evaluation

Evaluate all models on the untouched chronological test set.

### 9. Prediction logging and spatial analysis

Generate per-sample prediction logs and global performance maps.

### 10. XAI

Run attribution analysis and aggregate variable/pressure-level importance.

### 11. Runtime configuration

Generate and validate:

```text
channel_names.json
training_metadata.json
operational_thresholds.json
```

### 12. Historical verification

Run FNL-based verification against IBTrACS.

### 13. Post-classifier

Build the XAI/environment logistic candidate-triage model.

### 14. Active systems

Update `data/active_systems.json`.

### 15. Operational inference

Run `main.py` and inspect the generated GeoJSON/dashboard output.

---

## 14. Operational Example

Representative 24-hour inference:

```bash
python main.py   --once   --lead-times 24   --alert-threshold 0.75   --min-expert-confidence 0.45   --nms-radius-km 500   --max-detections 5   --classification-mode auto
```

Docker:

```bash
docker run --rm --gpus all   -v "$PWD:/workspace"   tc-dl-pipeline:latest   /workspace/main.py   --once   --lead-times 24   --alert-threshold 0.75   --min-expert-confidence 0.45   --nms-radius-km 500   --max-detections 5   --classification-mode auto
```

Institutional proxy settings, when required, should be supplied through the runtime environment. No proxy credentials are stored in the repository.

---

## 15. Reported Evaluation

For the reported 24-hour chronological holdout experiment on 2021–2022:

| Model | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Dual-Axis CNN-LSTM | 89.71% | 96.55% | 82.35% | 0.8889 | 0.9828 | 0.9818 |
| Conv-ViT-BiGRU | 92.28% | 98.32% | 86.03% | 0.9176 | 0.9699 | 0.9702 |
| Gated Ensemble | **93.01%** | 96.80% | **88.97%** | **0.9272** | 0.9752 | 0.9807 |

Ensemble confusion-matrix counts:

- TP: 121
- FP: 4
- FN: 15
- TN: 132

The ensemble therefore gives the strongest controlled holdout balance among the three reported models and produces the fewest missed genesis cases.

These metrics are specific to the documented chronological experiment. They should not be interpreted as a head-to-head comparison with operational NWP or AI weather models without matched dates, inputs, lead times, event definitions, sampling, and event-matching rules.

---

## 16. Limitations and Scientific Scope

This repository represents a research prototype for explainable global 24-hour genesis screening.

Important limitations:

1. Global scans retain a substantial false-alarm burden.
2. The XAI/environment post-classifier is trained on a relatively small retrospective detection set.
3. Operational input products can differ from the historical training product.
4. There is no matched head-to-head benchmark against ECMWF, GFS, or AI weather models in the present study.
5. XAI attribution should not automatically be interpreted as causal physical evidence.
6. More multi-year and basin-specific verification is required before operational decision use.

The intended role is automated early screening and prioritization for scientific/forecaster review, not replacement of operational tropical-cyclone guidance.

---

## 17. Data and Code Availability

### Code

https://github.com/Puranjay29/TC-Genesis-Research

A persistent archival DOI should be added after the corresponding GitHub release is deposited in Zenodo or another suitable repository.

### Data

The repository does not redistribute complete NCEP FNL or IBTrACS archives. Users should obtain the required datasets from their official archival providers and cite the exact versions used.

Large derived datasets and local caches are also excluded from Git.

---

## 18. Software Citation

Use the metadata in [`CITATION.cff`](CITATION.cff) when citing the software.

Once a persistent archive DOI is assigned, that DOI should be preferred for the exact software release.

---

## 19. Development and Reproducibility Notes

The cleaned repository intentionally excludes:

- raw GRIB archives;
- large extracted-data directories;
- local FNL caches;
- TensorBoard logs;
- transient forecast outputs;
- local credentials/tokens;
- historical development backups;
- unrelated legacy code.

The active `scripts/` layout is retained because `main.py` imports model builders and custom layers directly from that directory.

---

## 21. Contact

**Puranjay**  
USAR, GGSIPU  
New Delhi, India
