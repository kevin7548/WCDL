# Anomaly-based GRU Model for Korean Peninsula Winter Temperature Prediction

**[한국어 README](README_KR.md)**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

**Author:** Jongmin Kim
**Institution:** Seoul National University, Department of Earth and Environmental Sciences
**Project Type:** Undergraduate Thesis

---

## Overview

Predicting January mean temperature in the Korean Peninsula is crucial for **energy supply management** and **disaster response**. Traditional statistical and physical forecasting methods struggle to capture extreme cold waves and temperature anomalies.

This project applies a **GRU (Gated Recurrent Unit)** deep learning model to predict Korean winter temperature **anomalies** — deviations from the long-term monthly mean — using global climate indices and ERA5 reanalysis data spanning 45 years (1979-2023).

### Key Highlights

- **Anomaly-based approach**: removes the seasonal cycle so the model learns inter-annual variability, not just "summer is hot, winter is cold"
- **Proper ML methodology**: separate train/val/test splits, TimeSeriesSplit CV, no data leakage
- **Region-averaged ERA5 features**: area-weighted spatial means over physically meaningful regions instead of single grid-point selection
- **Baseline comparisons**: climatology, persistence, and Ridge regression baselines to verify GRU adds value
- **Production-ready code**: modular Python scripts with a single `run_pipeline.py` entry point

---

## Methodology

### Pipeline Overview

```
download -> preprocess -> select -> engineer -> train -> evaluate
```

Each step is independently runnable via `python run_pipeline.py --step <name>`.

### 1. Data Collection

| Source | Variables | Period |
|--------|-----------|--------|
| **NOAA PSL** | 23 Climate Indices (AO, NAO, ENSO, QBO, PDO, WP, etc.) | Monthly, 1979-2023 |
| **ERA5 Reanalysis** | MSLP, Sea Ice, OLR, 2m Temperature, Snow Cover | Monthly, 1979-2023 |

### 2. Preprocessing

- **Region-averaged ERA5 features**: area-weighted averages over physically meaningful regions (Arctic sea ice, Siberian High MSLP, Eurasian snow cover, etc.) instead of cherry-picking individual grid cells
- **Anomaly computation**: monthly climatology computed from **training data only** (1979-2014), then subtracted from all data to remove the seasonal cycle

### 3. Feature Selection

- Lag-correlation analysis on **anomaly** data across all months
- Forward-backward stepwise regression (p < 0.05)
- VIF-based multicollinearity removal (threshold = 10)

### 4. Feature Engineering

- Lagged features applied **once** (not double-lagged)
- GRU input shape: `(n_samples, 6, n_features)` — 6 timesteps of actual temporal sequence
- Scaling: StandardScaler fitted on training data only, separate scalers for X and y

### 5. Model Architecture

**GRU** was chosen for its efficiency with limited training data (fewer parameters than LSTM).

```
Input(6, n_features)
 -> GRU(64, return_sequences=True)
 -> Dropout(0.2)
 -> GRU(32, return_sequences=False)
 -> Dropout(0.2)
 -> Dense(1, linear)

Total parameters: ~25,000
```

**Training:**
- TimeSeriesSplit 5-fold cross-validation
- 3-way temporal split: train (1979-2014), val (2015-2019), test (2020-2023)
- EarlyStopping on validation loss (patience=20)
- TensorBoard experiment tracking

### 6. Baselines

| Model | Purpose |
|-------|---------|
| **Climatology** | Predicts mean training anomaly (trivial baseline) |
| **Persistence** | Predicts last observed anomaly |
| **Ridge Regression** | Linear model on same features |
| **GRU** | Non-linear temporal model |

---

## Project Structure

```
WCDL/
├── config.py                  # All constants, paths, hyperparameters
├── run_pipeline.py            # Main pipeline orchestrator
├── predict.py                 # Inference script
├── colab_runner.ipynb         # Thin Colab notebook (clone + run)
├── requirements.txt           # Dependencies
│
├── src/
│   ├── data/                  # Data download & loading
│   ├── preprocessing/         # Regridding, region-averaging, anomalies
│   ├── features/              # Feature selection & GRU sequence creation
│   ├── models/                # GRU builder, baselines, training loop
│   ├── evaluation/            # Metrics & plots
│   └── utils/                 # Scaling, path resolution
│
├── notebooks/                 # Original thesis notebooks (reference)
│   ├── 01_Data_Collection.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   └── 03_Modeling.ipynb
│
├── data/                      # Raw, interim, processed data
├── models/                    # Saved model weights
├── logs/                      # TensorBoard logs
└── outputs/                   # Result visualizations
```

---

## Quick Start

### Option A: Run on Google Colab (T4 GPU)

Open `colab_runner.ipynb` in Colab. It will clone the repo, install dependencies, and run the pipeline.

### Option B: Run Locally

```bash
# Clone & install
git clone https://github.com/kevin7548/WCDL.git
cd WCDL
pip install -r requirements.txt

# Run full pipeline (assumes data exists in data/)
python run_pipeline.py

# Or run individual steps
python run_pipeline.py --step preprocess
python run_pipeline.py --step train
python run_pipeline.py --step evaluate

# Predictions
python predict.py

# TensorBoard
tensorboard --logdir logs/
```

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.11 |
| **Deep Learning** | TensorFlow/Keras (GRU) |
| **Data Processing** | pandas, numpy, xarray, netCDF4 |
| **Visualization** | matplotlib, seaborn, Cartopy |
| **ML Utilities** | scikit-learn (StandardScaler, TimeSeriesSplit, Ridge) |
| **Statistics** | statsmodels (stepwise regression, VIF) |
| **Data Sources** | CDS API (ERA5), NOAA PSL |
| **Experiment Tracking** | TensorBoard |

---

## References

- Han, B., Lim, Y., Kim, H., & Son, S. (2018). "Monthly statistical prediction model for Korean Peninsula winter temperature." *Atmosphere*, 28(2), 153-162. DOI: 10.14191/Atmos.2018.28.2.153

---

## Contact

**Jongmin Kim** - Seoul National University
- GitHub: [@kevin7548](https://github.com/kevin7548)
- Email: jongmin.kim.k@gmail.com
