from __future__ import annotations

"""ERA5 spatial preprocessing: regridding and region-averaged indices.

Fixes Issue #1 (spatial data snooping):
  OLD: computed correlation at every grid point, then hand-picked
       the single grid cell with the highest correlation.
       With ~29,000 grid points this is extreme multiple testing.
  NEW: compute area-weighted averages over physically meaningful
       regions defined in ``config.ERA5_REGIONS``.

Refactored from notebook 02 cells 7-9, 21-22, 25, 27.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from config import ERA5_REGIONS

logger = logging.getLogger(__name__)


def regrid_era5(
    ds: xr.Dataset,
    target_resolution: float = 1.5,
) -> xr.Dataset:
    """Interpolate ERA5 data to a coarser regular lat-lon grid.

    Parameters
    ----------
    ds : xr.Dataset
        Raw ERA5 data with dims (time, lat, lon).
    target_resolution : float
        Target grid spacing in degrees.

    Returns
    -------
    xr.Dataset
        Regridded dataset.
    """
    new_lat = np.arange(-90.0, 90.0 + target_resolution, target_resolution)
    new_lon = np.arange(-180.0, 180.0 + target_resolution, target_resolution)

    ds_coarse = ds.interp(lat=new_lat, lon=new_lon, method="linear")
    logger.info(
        "Regridded ERA5: %d lat x %d lon (%.1f deg)",
        len(new_lat), len(new_lon), target_resolution,
    )
    return ds_coarse


def compute_region_average(
    ds_var: xr.DataArray,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
) -> pd.Series:
    """Compute an area-weighted spatial mean over a bounding box.

    Uses ``cos(lat)`` weighting to account for grid-cell area
    variations with latitude.

    Parameters
    ----------
    ds_var : xr.DataArray
        Variable with dims ``(time, lat, lon)``.
    lat_range : tuple
        (lat_min, lat_max) of the region.
    lon_range : tuple
        (lon_min, lon_max) of the region.

    Returns
    -------
    pd.Series
        Monthly time series of the region-averaged variable.
    """
    lat_min, lat_max = sorted(lat_range)
    lon_min, lon_max = sorted(lon_range)

    region = ds_var.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    if region.sizes.get("lat", 0) == 0 or region.sizes.get("lon", 0) == 0:
        raise ValueError(
            f"Empty region: lat={lat_range}, lon={lon_range}. "
            f"Available lat range: [{float(ds_var.lat.min())}, {float(ds_var.lat.max())}]"
        )

    weights = np.cos(np.deg2rad(region.lat))
    weighted_mean = region.weighted(weights).mean(dim=["lat", "lon"])

    return weighted_mean.to_series()


def compute_all_region_averages(
    ds: xr.Dataset,
    regions: dict | None = None,
) -> pd.DataFrame:
    """Compute area-weighted means for all configured regions.

    This replaces the single grid-point selection in the original code
    (e.g. ``mslp_var.sel(lat=27.0, lon=51.0)``).

    Parameters
    ----------
    ds : xr.Dataset
        ERA5 dataset with dims ``(time, lat, lon)``.
    regions : dict, optional
        Region definitions.  Defaults to ``config.ERA5_REGIONS``.

    Returns
    -------
    pd.DataFrame
        One column per region, DatetimeIndex.
    """
    regions = regions or ERA5_REGIONS
    series_dict: dict[str, pd.Series] = {}

    for region_name, cfg in regions.items():
        var_name = cfg["var"]
        lat_range = cfg["lat"]
        lon_range = cfg["lon"]

        if var_name not in ds.data_vars:
            logger.warning("Variable '%s' not in dataset, skipping region '%s'", var_name, region_name)
            continue

        logger.info("Computing region average: %s (%s)", region_name, cfg.get("description", ""))
        series = compute_region_average(ds[var_name], lat_range, lon_range)
        series.name = region_name
        series_dict[region_name] = series

    df = pd.DataFrame(series_dict)
    logger.info("Computed %d region-averaged features", len(df.columns))
    return df
