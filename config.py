"""Project-wide configuration constants for WCDL."""

from pathlib import Path
import os

# ─── Path Resolution (works on Colab + local) ───────────────────────
PROJECT_ROOT = Path(os.environ.get("WCDL_ROOT", Path(__file__).resolve().parent))
DATA_DIR = PROJECT_ROOT / "data"
RAW_ERA5_DIR = DATA_DIR / "raw" / "era5"
RAW_NOAA_DIR = DATA_DIR / "raw" / "noaa_indices"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ─── Time Ranges ────────────────────────────────────────────────────
TRAIN_START = "1979-01"
TRAIN_END = "2014-12"
VAL_START = "2015-01"
VAL_END = "2019-12"
TEST_START = "2020-01"
TEST_END = "2023-12"
FULL_START = "1979-01"
FULL_END = "2023-12"

N_CV_SPLITS = 5  # for TimeSeriesSplit

# ─── Target ─────────────────────────────────────────────────────────
TARGET_MONTH = 1  # January
TARGET_VAR = "t2m"
KOREA_LAT_RANGE = (33.0, 43.0)
KOREA_LON_RANGE = (124.0, 132.0)

# ─── ERA5 Region-Averaged Features ──────────────────────────────────
# Replaces single grid-point cherry-picking (Issue #1 fix).
# Each region is a physically meaningful area with established
# teleconnections to East Asian winter climate.
ERA5_REGIONS = {
    "arctic_sic": {
        "var": "icec",
        "lat": (65.0, 90.0),
        "lon": (-180.0, 180.0),
        "description": "Arctic sea-ice concentration",
    },
    "siberian_mslp": {
        "var": "mslp",
        "lat": (40.0, 60.0),
        "lon": (70.0, 120.0),
        "description": "Siberian High mean sea-level pressure",
    },
    "nao_iceland_mslp": {
        "var": "mslp",
        "lat": (60.0, 70.0),
        "lon": (-30.0, -10.0),
        "description": "Iceland Low MSLP (NAO node)",
    },
    "nao_azores_mslp": {
        "var": "mslp",
        "lat": (30.0, 40.0),
        "lon": (-30.0, -10.0),
        "description": "Azores High MSLP (NAO node)",
    },
    "eurasia_snowc": {
        "var": "snowc",
        "lat": (40.0, 70.0),
        "lon": (30.0, 150.0),
        "description": "Eurasian snow cover",
    },
    "tropical_pacific_olr": {
        "var": "mtnlwr",
        "lat": (-10.0, 10.0),
        "lon": (150.0, 270.0),
        "description": "Tropical Pacific OLR (convection proxy)",
    },
}

# ─── NOAA Climate Indices ───────────────────────────────────────────
NOAA_INDEX_NAMES = [
    "aao", "ammsst", "amon.us.long", "ao", "ea", "espi", "gmsst",
    "meiv2", "nao", "nina1.anom", "nina3.anom", "nina4.anom",
    "nina34.anom", "oni", "pacwarm", "pna", "qbo", "soi", "solar",
    "tna", "tsa", "whwp", "wp",
]

# ─── Feature Engineering ────────────────────────────────────────────
MAX_LAG_MONTHS = 6   # lookback window: 1..6 months
N_TIMESTEPS = 6      # GRU sequence length (== MAX_LAG_MONTHS)

# ─── Model Hyperparameters ──────────────────────────────────────────
GRU_UNITS = [64, 32]       # 2-layer GRU (~25K params vs old 1.78M)
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 20
REDUCE_LR_PATIENCE = 10
REDUCE_LR_FACTOR = 0.5

# ─── Baseline ───────────────────────────────────────────────────────
RIDGE_ALPHA = 1.0

# ─── Random Seed ────────────────────────────────────────────────────
SEED = 42
