from __future__ import annotations

"""Parse and clean NOAA PSL .data files.

This is a thin wrapper; the heavy lifting is in ``src.data.load``.
This module exists to hold any additional preprocessing steps
beyond raw parsing (e.g. index-specific transformations).
"""

import logging

import pandas as pd

from src.data.load import load_noaa_indices

logger = logging.getLogger(__name__)


def get_clean_indices(**kwargs) -> pd.DataFrame:
    """Load and return cleaned NOAA indices DataFrame.

    Any future index-specific transformations (unit conversions,
    detrending, etc.) should be added here.
    """
    df = load_noaa_indices(**kwargs)

    # Drop indices with excessive missing data (>10% NaN after parsing)
    frac_missing = df.isna().mean()
    bad = frac_missing[frac_missing > 0.10].index.tolist()
    if bad:
        logger.warning("Dropping indices with >10%% missing: %s", bad)
        df = df.drop(columns=bad)

    logger.info("Clean indices: %d variables, %d months", len(df.columns), len(df))
    return df
