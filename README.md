

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

> **GRU 기반 딥러닝 모델을 이용한 한반도 겨울철 기온 예측**
> GRU-based Deep Learning Model for Korean Peninsula Winter Temperature Prediction

**Author:** Jongmin Kim (김종민)
**Institution:** Seoul National University, Department of Earth and Environmental Sciences
**Project Type:** Undergraduate Thesis (졸업논문)

---

## Overview

Predicting January mean temperature in the Korean Peninsula is crucial for **energy supply management** and **disaster response**. Traditional statistical and physical forecasting methods struggle to capture extreme cold waves and temperature anomalies.

This project applies **GRU (Gated Recurrent Unit)** deep learning to improve prediction accuracy by learning non-linear relationships between global climate indices and Korean winter temperatures.

### Key Highlights

- Processed **45 years** of global climate data (1979-2023)
- Integrated **23 NOAA climate indices** + **5 ERA5 reanalysis variables**
- Achieved **RMSE ≈ 1K** for all-month predictions
- Applied **stepwise feature selection** with time-lag consideration

---

## Results

### Model Performance

| Metric | All Months | January Only |
|--------|------------|--------------|
| **Train RMSE** | 1.01 K | 1.41 K |
| **Test RMSE** | 1.19 K | 1.35 K |
| **Test R²** | 0.98 | -0.61 |

> **Note:** January-specific R² < 0 indicates the model struggles with extreme events. This limitation suggests potential for improvement with additional features or advanced architectures.

### Prediction Results

#### All Months (1979-2023)
![All Months Prediction](outputs/prediction_all_months.png)
*The model successfully captures seasonal temperature cycles with RMSE ≈ 1K*

#### January Prediction
![January Prediction](outputs/prediction_january.png)
*January predictions show the challenge of forecasting winter temperature extremes*

#### January Test Period (2015-2023)
![January Test Timeseries](outputs/january_test_timeseries.png)

#### Scatter Plot - January
![January Scatter](outputs/scatter_january.png)
*Predicted vs True temperature for January showing model performance*

---

## Methodology

### 1. Data Collection

| Source | Variables | Resolution |
|--------|-----------|------------|
| **NOAA PSL** | 23 Climate Indices (AO, NAO, ENSO, QBO, PDO, etc.) | Monthly |
| **ERA5 Reanalysis** | MSLP, Sea Ice, OLR, 2m Temperature, Snow Cover | 2.5° × 2.5° |

### 2. Feature Engineering

**Correlation Analysis with Time Lag:**
Computed correlations between each climate variable and Korean mean temperature considering lags of 1-6 months.

![Correlation Heatmap](outputs/correlation_heatmap_lag.png)
*Correlation coefficients between climate indices and Korean January temperature at different lag times*

### 3. Variable Selection

**Stepwise Regression:**
Selected optimal input variables that maximize adjusted R² while minimizing multicollinearity.

![Stepwise Selection](outputs/stepwise_selection.png)
*Adjusted R² progression during stepwise variable selection*

**Final Selected Features:**
| Variable | Description | Optimal Lag |
|----------|-------------|-------------|
| WP | Western Pacific Pattern | 6 months |
| WHWP | Western Hemisphere Warm Pool | 6 months |
| TSA | Tropical South Atlantic Index | 6 months |
| QBO | Quasi-Biennial Oscillation | 6 months |
| SNOWC_EURA | Eurasian Snow Cover | 6 months |

![Selected Features Heatmap](outputs/selected_features_heatmap.png)
*Correlation matrix of final selected features*

### 4. Model Architecture

**GRU (Gated Recurrent Unit)** was chosen over LSTM for:
- Fewer parameters → faster training
- Comparable performance on this dataset
- Better suited for limited training samples (36 years)

```
Model: Sequential
─────────────────────────────────────────────
Layer (type)              Output Shape    Param #
═════════════════════════════════════════════
GRU (return_seq=True)     (None, 1, 256)  199,680
Dropout (0.2)             (None, 1, 256)  0
GRU (return_seq=True)     (None, 1, 256)  394,752
Dropout (0.2)             (None, 1, 256)  0
GRU (return_seq=True)     (None, 1, 256)  394,752
Dropout (0.2)             (None, 1, 256)  0
GRU (return_seq=True)     (None, 1, 256)  394,752
Dropout (0.2)             (None, 1, 256)  0
GRU (return_seq=False)    (None, 256)     394,752
Dropout (0.2)             (None, 256)     0
Dense                     (None, 1)       257
═════════════════════════════════════════════
Total params: 1,778,945
─────────────────────────────────────────────
```

**Hyperparameter Tuning:**
5-fold cross-validation to find optimal configuration:
- Layers: 6 GRU layers
- Neurons: 256 per layer
- Batch size: 64
- Dropout: 0.2
- Early stopping: patience=15

---

## Project Structure

```
WCDL/
├── notebooks/
│   ├── 01_Data_Collection.ipynb      # ERA5 & NOAA data download
│   ├── 02_Data_Preprocessing.ipynb   # Feature engineering & correlation
│   └── 03_Modeling.ipynb             # GRU training & evaluation
│
├── data/
│   ├── raw/
│   │   ├── era5/                     # ERA5 reanalysis (.nc)
│   │   └── noaa_indices/             # Climate indices (.data)
│   ├── interim/                      # Intermediate files
│   └── processed/                    # Model-ready features
│
├── outputs/                          # Result visualizations
│   ├── correlation_heatmap_lag.png
│   ├── stepwise_selection.png
│   ├── prediction_all_months.png
│   ├── prediction_january.png
│   └── ...
│                         
└── README.md
```

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.11 |
| **Deep Learning** | TensorFlow, Keras (GRU, Dense, Dropout) |
| **Data Processing** | pandas, numpy, xarray, netCDF4 |
| **Visualization** | matplotlib, seaborn, Cartopy |
| **ML Utilities** | scikit-learn (MinMaxScaler, KFold, metrics) |
| **Statistics** | statsmodels (stepwise regression, VIF) |
| **Data Sources** | CDS API (ERA5), NOAA PSL |

---

## Getting Started

### Prerequisites

```bash
pip install tensorflow pandas numpy xarray netCDF4 matplotlib seaborn cartopy scikit-learn statsmodels cdsapi
```

### Running the Notebooks

1. **Data Collection** - Download climate data (requires CDS API key)
   ```bash
   jupyter notebook notebooks/01_Data_Collection.ipynb
   ```

2. **Preprocessing** - Feature engineering and correlation analysis
   ```bash
   jupyter notebook notebooks/02_Data_Preprocessing.ipynb
   ```

3. **Modeling** - Train GRU model and evaluate
   ```bash
   jupyter notebook notebooks/03_Modeling.ipynb
   ```

---

## Future Improvements

- [ ] Add more reanalysis variables (geopotential height, SST patterns)
- [ ] Experiment with attention mechanisms
- [ ] Implement ensemble methods combining multiple models
- [ ] Extend prediction to other winter months (December, February)

---

## References

- Han, B., Lim, Y., Kim, H., & Son, S. (2018). "Monthly statistical prediction model for Korean Peninsula winter temperature." *Atmosphere*, 28(2), 153-162. DOI: 10.14191/Atmos.2018.28.2.153

---

## Contact

**Jongmin Kim** - Seoul National University
- GitHub: [@kevin7548](https://github.com/kevin7548)
- Email: jongmin.kim.k@gmail.com
