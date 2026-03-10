"""Path resolution utilities for local and Colab environments."""

import os
import sys


def is_colab() -> bool:
    """Detect if running in Google Colab."""
    return "google.colab" in sys.modules


def ensure_dirs():
    """Create all required project directories if they don't exist."""
    from config import (
        DATA_DIR, RAW_ERA5_DIR, RAW_NOAA_DIR, INTERIM_DIR,
        PROCESSED_DIR, MODELS_DIR, LOGS_DIR, OUTPUTS_DIR,
    )

    for d in [
        DATA_DIR, RAW_ERA5_DIR, RAW_NOAA_DIR, INTERIM_DIR,
        PROCESSED_DIR, MODELS_DIR, LOGS_DIR, OUTPUTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
