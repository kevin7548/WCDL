# WCDL - Winter Climate Deep Learning

GRU-based deep learning model for predicting Korean Peninsula winter (January) temperature.

## Project Structure

```
WCDL/
├── notebooks/
│   ├── 01_Data_Collection.ipynb      # Download NOAA/ERA5 data
│   ├── 02_Data_Preprocessing.ipynb   # Feature engineering & correlation analysis
│   └── 03_Modeling.ipynb             # GRU model training & evaluation
│
├── data/
│   ├── raw/
│   │   ├── era5/                     # ERA5 reanalysis data (.nc)
│   │   └── noaa_indices/             # NOAA climate indices (.data)
│   ├── interim/                      # Intermediate processed files
│   └── processed/                    # Final features for modeling
│
├── models/                           # Saved model weights
└── results/figures/                  # Output plots
```

## Data Sources

- **NOAA Climate Indices**: 23 monthly indices (AO, NAO, ENSO, QBO, etc.)
- **ERA5 Reanalysis**: Sea ice, MSLP, OLR, 2m temperature, snow cover

## Model

- **Architecture**: GRU (Gated Recurrent Unit)
- **Hyperparameters**: 6 layers, 256 neurons, batch size 64, dropout 0.2
- **Training**: 1979-2014 (36 years)
- **Testing**: 2015-2023 (9 years)

## Selected Predictors

Final input variables selected via stepwise regression:
- WP (Western Pacific Pattern)
- WHWP (Western Hemisphere Warm Pool)
- TSA (Tropical South Atlantic Index)
- QBO (Quasi-Biennial Oscillation)
- SNOWC_EURA (Eurasian Snow Cover)
