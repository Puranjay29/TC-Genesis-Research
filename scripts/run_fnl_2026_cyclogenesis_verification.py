#!/usr/bin/env python3
"""
NCEP FNL Real-World Tropical Cyclogenesis Evaluation
================================================

Purpose
-------
Download native NCEP FNL GRIB2 analyses exactly one model lead time before observed tropical-cyclone genesis and evaluate them, reduce false positives scientifically,
and verify detections against IBTrACS genesis records.

Major improvements over the earlier script
-------------------------------------------
1. Exact 137-channel construction with strict level/variable validation.
2. Uses the trained 24/48/72-hour model only; no cross-horizon fallback.
3. Vectorized physical pre-screening before neural inference:
   - warm ocean
   - moist mid-troposphere
   - sufficient 850-hPa absolute vorticity
   - local surface-pressure minimum
   - ocean-only cells
4. Ensemble + CNN + ViT agreement requirement.
5. Great-circle clustering of neighbouring windows into one disturbance.
6. Maximum detections per atmospheric snapshot.
7. One joint Keras predictor where possible, avoiding three separate passes.
8. Matches each final disturbance only to IBTrACS genesis events occurring after initialization within the requested lead window.
9. Reports cyclone names, hits, misses, false alarms, POD, FAR and CSI.
10. Stores all outputs as CSV, GeoJSON and JSON summaries.

Expected data
-------------
GRIB files:
    /workspace/data/raw_2025_2026_grib/*.grib2

Models:
    /workspace/saved_models/best_cnn_lstm_24h.keras
    /workspace/saved_models/best_vit_gru_24h.keras
    /workspace/saved_models/best_ensemble_24h.keras
    /workspace/saved_models/channel_means.npy
    /workspace/saved_models/channel_stds.npy

IBTrACS:
    Pass --ibtracs-csv /workspace/data/ibtracs.ALL.list.v04r01.csv

Filename timestamp
------------------
The script expects a timestamp somewhere in each filename in a form such as:
    gfs_20240930_1800.grib2
    gfs_20240930_18.grib2
    2024093018.grib2

The valid forecast time is:
    analysis timestamp + --leadtime hours

Genesis definition
------------------
By default, genesis is the first valid IBTrACS track record for each SID.
Use --genesis-definition first-tropical to use the first record whose NATURE
is tropical/subtropical.

Example
-------
python3 /workspace/scripts/evaluate_realworld_fast.py \
    --leadtime 24 \
    --ibtracs-csv /workspace/data/ibtracs.ALL.list.v04r01.csv \
    --alert-threshold 0.80 \
    --min-expert-confidence 0.50 \
    --max-detections 5
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import logging
import math
import os
import re
import sys
import time
import requests
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from sklearn.cluster import DBSCAN
from tqdm import tqdm


# =============================================================================
# PATHS AND PROJECT IMPORTS
# =============================================================================

BASE_DIR = Path("/workspace")
SCRIPTS_DIR = BASE_DIR / "scripts"
GRIB_DIR = BASE_DIR / "data" / "raw_2025_2026_grib"
OUTPUT_DIR = BASE_DIR / "data" / "extracted_features_real_world"
MODELS_DIR = BASE_DIR / "saved_models"
FNL_CACHE_DIR = BASE_DIR / "data" / "fnl_2026_grib"
FNL_HTTP_ROOT = (
    "https://thredds.rda.ucar.edu/thredds/fileServer/"
    "files/g/d083002/grib2"
)

for directory in (OUTPUT_DIR, MODELS_DIR, FNL_CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

for path in (str(SCRIPTS_DIR), str(BASE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from train_ensemble import build_advanced_ensemble
except ImportError as exc:
    raise ImportError(
        "Could not import build_advanced_ensemble from "
        "/workspace/scripts/train_ensemble.py"
    ) from exc


# =============================================================================
# LOGGING / TENSORFLOW
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    stream=sys.stdout,
)

tf.config.optimizer.set_experimental_options({"layout_optimizer": False})


# =============================================================================
# CHANNEL SCHEMA
# =============================================================================

V3D_NAMES: Tuple[str, ...] = (
    "ugrdprs",
    "vgrdprs",
    "vvelprs",
    "tmpprs",
    "rhprs",
    "hgtprs",
    "absvprs",
)

V2D_NAMES: Tuple[str, ...] = (
    "pressfc",
    "capesfc",
    "tmpsfc",
    "landmask",
)

PRESSURE_LEVELS: Tuple[float, ...] = (
    1000.0, 975.0, 950.0, 925.0, 900.0, 850.0, 800.0, 750.0,
    700.0, 650.0, 600.0, 550.0, 500.0, 450.0, 400.0, 350.0,
    300.0, 250.0, 200.0,
)

PATCH_SIZE = 21
HALF_PATCH = PATCH_SIZE // 2
NUM_CHANNELS = len(V3D_NAMES) * len(PRESSURE_LEVELS) + len(V2D_NAMES)
EARTH_RADIUS_KM = 6371.0088

if NUM_CHANNELS != 137:
    raise RuntimeError(f"Internal channel schema error: {NUM_CHANNELS}")


@dataclass(frozen=True)
class Config:
    leadtime: int
    threshold: float
    min_expert_confidence: float
    batch_size: int
    min_latitude: float
    max_latitude: float
    maximum_land_fraction: float
    minimum_sst_k: float
    minimum_midlevel_rh_percent: float
    minimum_vorticity_850_s1: float
    maximum_pressure_anomaly_pa: float
    maximum_shear_200_850_ms: float
    maximum_lowlevel_divergence_s1: float
    cluster_radius_km: float
    min_cluster_support: int
    max_detections: int
    lead_tolerance_hours: float
    ibtracs_distance_tolerance_km: float
    genesis_definition: str


# =============================================================================
# BASIC GEODESY
# =============================================================================

def normalize_longitude_180(longitude: float) -> float:
    value = ((float(longitude) + 180.0) % 360.0) - 180.0
    return 180.0 if math.isclose(value, -180.0) else value


def normalize_longitude_360(longitude: np.ndarray) -> np.ndarray:
    return np.mod(longitude.astype(np.float64), 360.0)


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(normalize_longitude_180(lon2 - lon1))

    value = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(value)))


# =============================================================================
# FILE TIME
# =============================================================================

_TIMESTAMP_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(?P<date>\d{8})[_-](?P<hour>\d{4})"),
    re.compile(r"(?P<date>\d{8})[_-](?P<hour>\d{2})(?!\d)"),
    re.compile(r"(?P<datehour>\d{10})(?!\d)"),
)


def parse_analysis_time(path: Path) -> datetime:
    name = path.name

    for pattern in _TIMESTAMP_PATTERNS:
        match = pattern.search(name)
        if not match:
            continue

        groups = match.groupdict()

        if groups.get("datehour"):
            dt = datetime.strptime(groups["datehour"], "%Y%m%d%H")
            return dt.replace(tzinfo=timezone.utc)

        date_text = groups["date"]
        hour_text = groups["hour"]
        hour = int(hour_text[:2])
        dt = datetime.strptime(date_text, "%Y%m%d").replace(hour=hour)
        return dt.replace(tzinfo=timezone.utc)

    raise ValueError(
        f"Could not parse YYYYMMDD/hour timestamp from filename: {name}"
    )



# =============================================================================
# NCEP FNL DOWNLOAD
# =============================================================================

def fnl_filename(analysis_time: datetime) -> str:
    return analysis_time.strftime("fnl_%Y%m%d_%H_00.grib2")


def fnl_download_url(analysis_time: datetime) -> str:
    year = analysis_time.strftime("%Y")
    year_month = analysis_time.strftime("%Y.%m")
    return (
        f"{FNL_HTTP_ROOT}/{year}/{year_month}/"
        f"{fnl_filename(analysis_time)}"
    )


def download_fnl_file(
    analysis_time: datetime,
    cache_dir: Path,
    *,
    timeout_seconds: int = 180,
    retries: int = 4,
    insecure_ssl: bool = False,
) -> Path:
    """
    Download the native 1-degree, six-hourly NCEP FNL GRIB2 analysis from
    NCAR GDEX/THREDDS. Existing non-empty files are reused.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / fnl_filename(analysis_time)

    if destination.exists() and destination.stat().st_size > 1_000_000:
        logging.info("Using cached FNL file: %s", destination)
        return destination

    url = fnl_download_url(analysis_time)
    temporary = destination.with_suffix(destination.suffix + ".part")

    headers = {
        "User-Agent": (
            "TC-Genesis-Research/1.0 "
            "(NCEP FNL scientific verification)"
        )
    }

    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            logging.info(
                "Downloading FNL %s (attempt %d/%d)",
                analysis_time.isoformat(),
                attempt,
                retries,
            )
            with requests.get(
                url,
                stream=True,
                timeout=timeout_seconds,
                headers=headers,
                verify=not insecure_ssl,
            ) as response:
                if response.status_code == 404:
                    raise FileNotFoundError(
                        f"FNL archive file not available: {url}"
                    )
                response.raise_for_status()

                content_length = int(
                    response.headers.get("content-length", "0") or "0"
                )

                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if chunk:
                            handle.write(chunk)

                downloaded_size = temporary.stat().st_size
                if downloaded_size < 1_000_000:
                    raise IOError(
                        f"Downloaded FNL file is unexpectedly small: "
                        f"{downloaded_size} bytes"
                    )
                if content_length and downloaded_size != content_length:
                    raise IOError(
                        f"Incomplete FNL download: {downloaded_size} of "
                        f"{content_length} bytes"
                    )

                temporary.replace(destination)
                logging.info(
                    "Saved FNL file: %s (%.1f MB)",
                    destination,
                    destination.stat().st_size / (1024 ** 2),
                )
                return destination

        except Exception as exc:
            last_error = exc
            if temporary.exists():
                temporary.unlink()
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))

    raise RuntimeError(
        f"Failed to download FNL for {analysis_time.isoformat()}: "
        f"{last_error}"
    )


def select_fnl_analysis_times(
    genesis_catalog: pd.DataFrame,
    *,
    start_date: datetime,
    end_date: datetime,
    leadtime_hours: int,
) -> List[datetime]:
    """
    Build one exact pre-genesis FNL initialization per observed cyclone.

    IBTrACS records are six-hourly, and FNL is available at 00/06/12/18 UTC,
    so genesis_time - leadtime should align with an FNL analysis cycle.
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    selected = genesis_catalog[
        (genesis_catalog["genesis_time"] >= start_ts)
        & (genesis_catalog["genesis_time"] < end_ts)
    ].copy()

    analysis_times: set[datetime] = set()

    for genesis_time in selected["genesis_time"]:
        analysis = (
            pd.Timestamp(genesis_time)
            - pd.Timedelta(hours=leadtime_hours)
        )

        if analysis.minute != 0 or analysis.hour not in {0, 6, 12, 18}:
            logging.warning(
                "Skipping non-synoptic analysis time derived from genesis: %s",
                analysis,
            )
            continue

        analysis_times.add(analysis.to_pydatetime())

    return sorted(analysis_times)


# =============================================================================
# GRIB EXTRACTION
# =============================================================================

def load_filtered_grib(
    path: Path,
    *,
    short_name: str,
    type_of_level: str,
    extra_keys: Optional[Mapping[str, Any]] = None,
) -> xr.Dataset:
    keys: Dict[str, Any] = {
        "shortName": short_name,
        "typeOfLevel": type_of_level,
    }
    if extra_keys:
        keys.update(extra_keys)

    return xr.load_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "errors": "raise",
            "indexpath": "",
            "filter_by_keys": keys,
        },
    )


def one_data_array(ds: xr.Dataset, description: str) -> xr.DataArray:
    variables = list(ds.data_vars)
    if len(variables) != 1:
        raise ValueError(
            f"{description}: expected one data variable, found {variables}"
        )
    return ds[variables[0]]


def standardize_coordinates(da: xr.DataArray) -> xr.DataArray:
    rename: Dict[str, str] = {}
    for old, new in (
        ("latitude", "lat"),
        ("longitude", "lon"),
        ("isobaricInhPa", "lev"),
    ):
        if old in da.dims or old in da.coords:
            rename[old] = new

    if rename:
        da = da.rename(rename)

    if "lat" not in da.coords or "lon" not in da.coords:
        raise ValueError(f"Missing lat/lon in {da.name}")

    scalar_coords = [
        name for name, coord in da.coords.items()
        if coord.ndim == 0 and name not in {"lat", "lon", "lev"}
    ]
    if scalar_coords:
        da = da.drop_vars(scalar_coords)

    if da["lat"].values[0] > da["lat"].values[-1]:
        da = da.sortby("lat")

    new_lon = normalize_longitude_360(da["lon"].values)
    da = da.assign_coords(lon=new_lon).sortby("lon")

    unique_lon, indices = np.unique(da["lon"].values, return_index=True)
    if len(unique_lon) != da.sizes["lon"]:
        da = da.isel(lon=np.sort(indices))

    return da


def harmonize_units(variable: str, da: xr.DataArray) -> xr.DataArray:
    units = str(da.attrs.get("units", "")).lower().strip()
    result = da.astype(np.float32)

    if variable == "rhprs":
        maximum = float(result.max(skipna=True).values)
        if units in {"1", "fraction"} or maximum <= 1.5:
            result = result * 100.0

    elif variable == "pressfc":
        if units in {"hpa", "mb", "mbar", "millibar"}:
            result = result * 100.0

    elif variable == "hgtprs":
        if "m**2" in units or "m2 s-2" in units or "m^2 s^-2" in units:
            result = result / np.float32(9.80665)

    elif variable == "landmask":
        result = result.clip(min=0.0, max=1.0)

    return result


PRESSURE_SHORT_NAMES: Mapping[str, str] = {
    "ugrdprs": "u",
    "vgrdprs": "v",
    "vvelprs": "w",
    "tmpprs": "t",
    "rhprs": "r",
    "hgtprs": "gh",
    "absvprs": "absv",
}

SURFACE_CANDIDATES: Mapping[str, Sequence[Mapping[str, Any]]] = {
    "pressfc": (
        {"shortName": "sp", "typeOfLevel": "surface"},
        {"shortName": "pres", "typeOfLevel": "surface"},
    ),
    "capesfc": (
        {"shortName": "cape", "typeOfLevel": "surface"},
        {"shortName": "cape", "typeOfLevel": "heightAboveGround", "level": 0},
        {"shortName": "cape", "typeOfLevel": "pressureFromGroundLayer"},
    ),
    "tmpsfc": (
        {"shortName": "t", "typeOfLevel": "surface"},
        {"shortName": "skt", "typeOfLevel": "surface"},
    ),
    "landmask": (
        {"shortName": "lsm", "typeOfLevel": "surface"},
        {"shortName": "land", "typeOfLevel": "surface"},
    ),
}


def extract_pressure(path: Path, project_name: str) -> xr.DataArray:
    short_name = PRESSURE_SHORT_NAMES[project_name]
    ds = load_filtered_grib(
        path,
        short_name=short_name,
        type_of_level="isobaricInhPa",
    )
    da = standardize_coordinates(
        one_data_array(ds, f"{project_name}/{short_name}")
    )

    if "lev" not in da.coords:
        raise ValueError(f"{project_name} has no pressure levels")

    available = {float(value) for value in da["lev"].values}
    missing = [level for level in PRESSURE_LEVELS if level not in available]
    if missing:
        raise ValueError(f"{project_name} missing levels: {missing}")

    da = da.sel(lev=list(PRESSURE_LEVELS)).transpose("lev", "lat", "lon")
    da = harmonize_units(project_name, da)
    da.name = project_name
    return da


def extract_surface(path: Path, project_name: str) -> xr.DataArray:
    errors: List[str] = []

    for candidate in SURFACE_CANDIDATES[project_name]:
        extra = {
            key: value
            for key, value in candidate.items()
            if key not in {"shortName", "typeOfLevel"}
        }
        try:
            ds = load_filtered_grib(
                path,
                short_name=str(candidate["shortName"]),
                type_of_level=str(candidate["typeOfLevel"]),
                extra_keys=extra,
            )
            da = standardize_coordinates(
                one_data_array(ds, f"{project_name}/{candidate}")
            )

            for dim in list(da.dims):
                if dim not in {"lat", "lon"}:
                    if da.sizes[dim] != 1:
                        raise ValueError(
                            f"Unexpected {project_name} dimension "
                            f"{dim}={da.sizes[dim]}"
                        )
                    da = da.squeeze(dim, drop=True)

            da = da.transpose("lat", "lon")
            da = harmonize_units(project_name, da)
            da.name = project_name
            return da

        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    raise ValueError(
        f"Could not extract {project_name}. Attempts: {' | '.join(errors)}"
    )


def preprocess_grib(path: Path) -> xr.Dataset:
    arrays: List[xr.DataArray] = []

    for variable in V3D_NAMES:
        arrays.append(extract_pressure(path, variable))

    for variable in V2D_NAMES:
        arrays.append(extract_surface(path, variable))

    ds = xr.merge(arrays, join="exact", compat="override")

    target_lats = np.arange(-90.0, 91.0, 1.0, dtype=np.float32)
    target_lons = np.arange(0.0, 360.0, 1.0, dtype=np.float32)

    ds = ds.interp(
        lat=target_lats,
        lon=target_lons,
        method="linear",
    )

    for variable in list(V3D_NAMES) + list(V2D_NAMES):
        if not np.all(np.isfinite(ds[variable].values)):
            raise ValueError(
                f"{path.name}: {variable} contains missing/non-finite values"
            )

    return ds


# =============================================================================
# NORMALIZATION AND MODEL
# =============================================================================

def load_normalization() -> Tuple[np.ndarray, np.ndarray]:
    means = np.asarray(
        np.load(MODELS_DIR / "channel_means.npy"),
        dtype=np.float32,
    ).reshape(-1)
    stds = np.asarray(
        np.load(MODELS_DIR / "channel_stds.npy"),
        dtype=np.float32,
    ).reshape(-1)

    # Project convention: 136 standardized atmospheric channels and identity
    # normalization for the final landmask channel.
    if means.shape == (136,) and stds.shape == (136,):
        means = np.concatenate([means, np.array([0.0], dtype=np.float32)])
        stds = np.concatenate([stds, np.array([1.0], dtype=np.float32)])

    if means.shape != (137,) or stds.shape != (137,):
        raise ValueError(
            f"Expected 137 stats or 136 + landmask identity; "
            f"got means={means.shape}, stds={stds.shape}"
        )

    if np.any(~np.isfinite(means)) or np.any(~np.isfinite(stds)):
        raise ValueError("Normalization statistics contain NaN/Inf")
    if np.any(stds <= 0):
        raise ValueError("Normalization standard deviations must be > 0")

    return means, stds


def construct_grid(
    ds: xr.Dataset,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    layers: List[np.ndarray] = []
    index = 0

    for variable in V3D_NAMES:
        for level in PRESSURE_LEVELS:
            values = np.asarray(
                ds[variable].sel(lev=level).values,
                dtype=np.float32,
            )
            layers.append((values - means[index]) / stds[index])
            index += 1

    for variable in V2D_NAMES:
        values = np.asarray(ds[variable].values, dtype=np.float32)
        layers.append((values - means[index]) / stds[index])
        index += 1

    grid = np.stack(layers, axis=-1).astype(np.float32)

    if grid.shape != (181, 360, 137):
        raise ValueError(f"Unexpected model grid shape: {grid.shape}")

    return grid


def load_models(
    leadtime: int,
) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    cnn_path = MODELS_DIR / f"best_cnn_lstm_{leadtime}h.keras"
    vit_path = MODELS_DIR / f"best_vit_gru_{leadtime}h.keras"
    ensemble_path = MODELS_DIR / f"best_ensemble_{leadtime}h.keras"

    missing = [
        path for path in (cnn_path, vit_path, ensemble_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing lead-time-specific models: "
            + ", ".join(str(path) for path in missing)
        )

    ensemble, cnn, vit = build_advanced_ensemble(
        str(cnn_path),
        str(vit_path),
        input_shape=(21, 21, 137),
    )
    ensemble.load_weights(str(ensemble_path))

    # A single model call is substantially faster than three predict() calls.
    try:
        joint = tf.keras.Model(
            inputs=ensemble.input,
            outputs=[ensemble.output, cnn.output, vit.output],
            name=f"joint_predictor_{leadtime}h",
        )
        _ = joint(
            tf.zeros((1, 21, 21, 137), dtype=tf.float32),
            training=False,
        )
        logging.info("Using one-pass joint ensemble/CNN/ViT predictor.")
    except Exception as exc:
        logging.warning(
            "Could not create joint predictor (%s); using a wrapper.",
            exc,
        )

        class SequentialWrapper(tf.keras.Model):
            def call(self, inputs, training=False):
                return (
                    ensemble(inputs, training=training),
                    cnn(inputs, training=training),
                    vit(inputs, training=training),
                )

        joint = SequentialWrapper()

    return ensemble, cnn, vit, joint


# =============================================================================
# SCIENTIFIC CANDIDATE SCREENING
# =============================================================================

def pressure_anomaly_field(
    pressure: np.ndarray,
    radius: int = 3,
) -> np.ndarray:
    """
    Local centre pressure minus the surrounding 7 x 7 mean.

    Longitude is cyclic. Latitude is edge-padded.
    """
    padded = np.pad(
        pressure,
        ((radius, radius), (radius, radius)),
        mode="wrap",
    )

    # Correct latitude edges after cyclic padding by replacing them with edge
    # replication. Longitude remains cyclic.
    padded[:radius, :] = padded[radius:radius + 1, :]
    padded[-radius:, :] = padded[-radius - 1:-radius, :]

    integral = np.pad(
        padded,
        ((1, 0), (1, 0)),
        mode="constant",
        constant_values=0,
    ).cumsum(0).cumsum(1)

    size = 2 * radius + 1
    window_sum = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )

    local_mean = window_sum / float(size * size)
    return pressure - local_mean


def candidate_indices(
    ds: xr.Dataset,
    config: Config,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    lat = np.asarray(ds["lat"].values, dtype=np.float32)
    lon = np.asarray(ds["lon"].values, dtype=np.float32)
    land = np.asarray(ds["landmask"].values, dtype=np.float32)
    sst = np.asarray(ds["tmpsfc"].values, dtype=np.float32)
    pressure = np.asarray(ds["pressfc"].values, dtype=np.float32)

    rh_mid = np.asarray(
        ds["rhprs"].sel(lev=[700.0, 600.0, 500.0]).mean("lev").values,
        dtype=np.float32,
    )

    raw_vort_850 = np.asarray(
        ds["absvprs"].sel(lev=850.0).values,
        dtype=np.float32,
    )

    # Hemisphere-aware cyclonic sign:
    # positive in the Northern Hemisphere and negative in the Southern.
    hemisphere_sign = np.where(lat[:, None] >= 0.0, 1.0, -1.0)
    cyclonic_vort_850 = raw_vort_850 * hemisphere_sign

    u850 = np.asarray(
        ds["ugrdprs"].sel(lev=850.0).values,
        dtype=np.float32,
    )
    v850 = np.asarray(
        ds["vgrdprs"].sel(lev=850.0).values,
        dtype=np.float32,
    )
    u200 = np.asarray(
        ds["ugrdprs"].sel(lev=200.0).values,
        dtype=np.float32,
    )
    v200 = np.asarray(
        ds["vgrdprs"].sel(lev=200.0).values,
        dtype=np.float32,
    )

    shear_200_850 = np.sqrt(
        (u200 - u850) ** 2 + (v200 - v850) ** 2
    ).astype(np.float32)

    # Approximate horizontal divergence at 850 hPa on the regular lat/lon grid.
    # Longitude is cyclic; latitude uses NumPy edge gradients.
    earth_radius_m = 6_371_008.8
    lat_rad = np.radians(lat.astype(np.float64))
    lon_rad = np.radians(lon.astype(np.float64))

    du_dlambda = np.gradient(
        u850.astype(np.float64),
        lon_rad,
        axis=1,
        edge_order=1,
    )
    dv_dphi = np.gradient(
        v850.astype(np.float64),
        lat_rad,
        axis=0,
        edge_order=1,
    )

    cos_lat = np.clip(np.cos(lat_rad), 0.10, None)[:, None]
    lowlevel_divergence = (
        du_dlambda / (earth_radius_m * cos_lat)
        + dv_dphi / earth_radius_m
    ).astype(np.float32)

    pressure_anomaly = pressure_anomaly_field(pressure, radius=3)

    lat_mask = (
        (lat >= config.min_latitude)
        & (lat <= config.max_latitude)
    )[:, None]

    valid_lat_patch = np.zeros_like(lat_mask, dtype=bool)
    valid_lat_patch[HALF_PATCH:-HALF_PATCH, 0] = True

    mask = (
        lat_mask
        & valid_lat_patch
        & (land <= config.maximum_land_fraction)
        & (sst >= config.minimum_sst_k)
        & (rh_mid >= config.minimum_midlevel_rh_percent)
        & (cyclonic_vort_850 >= config.minimum_vorticity_850_s1)
        & (pressure_anomaly <= config.maximum_pressure_anomaly_pa)
        & (shear_200_850 <= config.maximum_shear_200_850_ms)
        & (
            lowlevel_divergence
            <= config.maximum_lowlevel_divergence_s1
        )
    )

    indices = np.argwhere(mask)

    diagnostics = {
        "lat": lat,
        "lon": lon,
        "land": land,
        "sst": sst,
        "rh_mid": rh_mid,
        "vort_850": cyclonic_vort_850,
        "raw_vort_850": raw_vort_850,
        "pressure_anomaly": pressure_anomaly,
        "shear_200_850": shear_200_850,
        "lowlevel_divergence": lowlevel_divergence,
    }
    return indices, diagnostics



def extract_patch_batch(
    padded_grid: np.ndarray,
    batch_indices: np.ndarray,
) -> np.ndarray:
    patches = np.empty(
        (len(batch_indices), PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS),
        dtype=np.float32,
    )

    for position, (lat_index, lon_index) in enumerate(batch_indices):
        padded_lon = int(lon_index) + HALF_PATCH
        patches[position] = padded_grid[
            int(lat_index) - HALF_PATCH:int(lat_index) + HALF_PATCH + 1,
            padded_lon - HALF_PATCH:padded_lon + HALF_PATCH + 1,
            :,
        ]

    return patches


def predict_joint(
    joint_model: tf.keras.Model,
    x_batch: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ensemble, cnn, vit = joint_model(
        tf.convert_to_tensor(x_batch),
        training=False,
    )
    return (
        np.asarray(ensemble).reshape(-1),
        np.asarray(cnn).reshape(-1),
        np.asarray(vit).reshape(-1),
    )


# =============================================================================
# CLUSTERING
# =============================================================================

def cluster_detections(
    raw: Sequence[Dict[str, Any]],
    config: Config,
) -> List[Dict[str, Any]]:
    if not raw:
        return []

    coordinates_rad = np.radians(
        np.array(
            [[item["latitude"], item["longitude"]] for item in raw],
            dtype=np.float64,
        )
    )

    labels = DBSCAN(
        eps=config.cluster_radius_km / EARTH_RADIUS_KM,
        min_samples=config.min_cluster_support,
        metric="haversine",
        algorithm="ball_tree",
    ).fit_predict(coordinates_rad)

    clusters: List[Dict[str, Any]] = []

    for cluster_label in sorted(set(labels)):
        if cluster_label < 0:
            continue

        members = [
            raw[index]
            for index in np.where(labels == cluster_label)[0]
        ]

        best = max(
            members,
            key=lambda item: float(item["confidence"]),
        ).copy()

        confidences = np.array(
            [item["confidence"] for item in members],
            dtype=np.float64,
        )

        weights = np.maximum(confidences, 1e-6)
        latitudes = np.array(
            [item["latitude"] for item in members],
            dtype=np.float64,
        )
        longitudes_rad = np.radians(
            [item["longitude"] for item in members]
        )

        weighted_latitude = float(
            np.average(latitudes, weights=weights)
        )
        weighted_x = float(
            np.average(np.cos(longitudes_rad), weights=weights)
        )
        weighted_y = float(
            np.average(np.sin(longitudes_rad), weights=weights)
        )
        weighted_longitude = normalize_longitude_180(
            math.degrees(math.atan2(weighted_y, weighted_x))
        )

        best["latitude"] = weighted_latitude
        best["longitude"] = weighted_longitude
        best["representative_grid_latitude"] = float(
            max(members, key=lambda item: item["confidence"])["latitude"]
        )
        best["representative_grid_longitude"] = float(
            max(members, key=lambda item: item["confidence"])["longitude"]
        )
        best["cluster_support"] = len(members)
        best["cluster_mean_confidence"] = float(confidences.mean())
        best["cluster_max_confidence"] = float(confidences.max())
        # Combined scientific ranking. Neural confidence remains dominant,
        # while coherent support and favourable physical structure break ties.
        ensemble_score = float(best["confidence"])
        cnn_score = float(best["cnn_confidence"])
        vit_score = float(best["vit_confidence"])
        support_score = min(1.0, math.log1p(len(members)) / math.log(11.0))
        vort_score = min(
            1.0,
            max(0.0, float(best["vorticity_850_s1"])) / 5.0e-5,
        )
        rh_score = min(
            1.0,
            max(0.0, float(best["midlevel_rh_percent"]) - 50.0) / 30.0,
        )
        pressure_score = min(
            1.0,
            max(0.0, -float(best["pressure_anomaly_pa"])) / 500.0,
        )
        convergence_score = min(
            1.0,
            max(0.0, -float(best["lowlevel_divergence_s1"])) / 5.0e-6,
        )
        shear_score = min(
            1.0,
            max(0.0, 20.0 - float(best["shear_200_850_ms"])) / 20.0,
        )

        best["cluster_rank_score"] = float(
            0.44 * ensemble_score
            + 0.14 * cnn_score
            + 0.14 * vit_score
            + 0.08 * support_score
            + 0.06 * vort_score
            + 0.04 * rh_score
            + 0.04 * pressure_score
            + 0.03 * convergence_score
            + 0.03 * shear_score
        )

        clusters.append(best)

    clusters.sort(
        key=lambda item: item["cluster_rank_score"],
        reverse=True,
    )
    return clusters[:config.max_detections]


# =============================================================================
# IBTRACS
# =============================================================================

def read_ibtracs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"IBTrACS CSV not found: {path}")

    # IBTrACS list files commonly contain a units row directly after headers.
    frame = pd.read_csv(path, low_memory=False)

    required = {"SID", "NAME", "ISO_TIME", "LAT", "LON"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"IBTrACS missing columns {sorted(missing)}. "
            f"Available columns include: {list(frame.columns)[:30]}"
        )

    frame["ISO_TIME"] = pd.to_datetime(
        frame["ISO_TIME"],
        errors="coerce",
        utc=True,
    )
    frame["LAT"] = pd.to_numeric(frame["LAT"], errors="coerce")
    frame["LON"] = pd.to_numeric(frame["LON"], errors="coerce")

    frame = frame.dropna(
        subset=["SID", "ISO_TIME", "LAT", "LON"]
    ).copy()

    frame["NAME"] = (
        frame["NAME"]
        .fillna("UNNAMED")
        .astype(str)
        .str.strip()
        .replace({"": "UNNAMED", "NOT_NAMED": "UNNAMED"})
    )

    return frame


def build_genesis_catalog(
    ibtracs: pd.DataFrame,
    definition: str,
) -> pd.DataFrame:
    frame = ibtracs.sort_values(["SID", "ISO_TIME"]).copy()

    if definition == "first-tropical" and "NATURE" in frame.columns:
        tropical_natures = {
            "TS", "SS", "ET", "MX", "NR", "DS",
        }
        nature = frame["NATURE"].fillna("").astype(str).str.strip()
        filtered = frame[nature.isin(tropical_natures)].copy()

        # Storms without a qualifying NATURE record fall back to first track.
        first_tropical = filtered.groupby("SID", as_index=False).first()
        existing = set(first_tropical["SID"])
        fallback = (
            frame[~frame["SID"].isin(existing)]
            .groupby("SID", as_index=False)
            .first()
        )
        genesis = pd.concat(
            [first_tropical, fallback],
            ignore_index=True,
        )
    else:
        genesis = frame.groupby("SID", as_index=False).first()

    genesis = genesis.rename(
        columns={
            "ISO_TIME": "genesis_time",
            "LAT": "genesis_latitude",
            "LON": "genesis_longitude",
            "NAME": "cyclone_name",
        }
    )

    keep = [
        column for column in (
            "SID",
            "cyclone_name",
            "genesis_time",
            "genesis_latitude",
            "genesis_longitude",
            "BASIN",
            "SUBBASIN",
            "SEASON",
        )
        if column in genesis.columns
    ]
    return genesis[keep].copy()


def candidate_events_for_analysis_time(
    genesis_catalog: pd.DataFrame,
    analysis_time: datetime,
    target_lead_hours: float,
    lead_tolerance_hours: float,
) -> pd.DataFrame:
    """
    Select only true pre-genesis events.

    A tropical-cyclogenesis forecast is valid only when genesis occurs after
    model initialization. For a nominal 24-hour forecast and a 6-hour
    tolerance, the accepted lead window is 18-30 hours after initialization.

    This deliberately rejects:
    - genesis before initialization,
    - genesis exactly at initialization,
    - post-genesis cyclone detections,
    - events far outside the requested lead-time window.
    """
    analysis_timestamp = pd.Timestamp(analysis_time)

    actual_lead_hours = (
        genesis_catalog["genesis_time"] - analysis_timestamp
    ).dt.total_seconds() / 3600.0

    minimum_lead = max(0.0, target_lead_hours - lead_tolerance_hours)
    maximum_lead = target_lead_hours + lead_tolerance_hours

    mask = (
        (actual_lead_hours >= minimum_lead)
        & (actual_lead_hours <= maximum_lead)
    )

    events = genesis_catalog[mask].copy()
    events["actual_lead_hours"] = actual_lead_hours[mask].values
    events["lead_error_hours"] = (
        events["actual_lead_hours"] - target_lead_hours
    ).abs()

    return events



def match_predictions_to_events(
    predictions: List[Dict[str, Any]],
    events: pd.DataFrame,
    config: Config,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Greedy one-to-one spatial matching by smallest distance.

    Returns:
        annotated predictions
        event records with hit/miss status
    """
    annotated = [item.copy() for item in predictions]

    event_records = events.to_dict("records")
    used_predictions: set[int] = set()
    used_events: set[int] = set()

    candidate_pairs: List[Tuple[float, int, int]] = []

    for prediction_index, prediction in enumerate(annotated):
        for event_index, event in enumerate(event_records):
            distance = haversine_km(
                float(prediction["latitude"]),
                float(prediction["longitude"]),
                float(event["genesis_latitude"]),
                float(event["genesis_longitude"]),
            )
            if distance <= config.ibtracs_distance_tolerance_km:
                candidate_pairs.append(
                    (distance, prediction_index, event_index)
                )

    for distance, prediction_index, event_index in sorted(candidate_pairs):
        if prediction_index in used_predictions or event_index in used_events:
            continue

        prediction = annotated[prediction_index]
        event = event_records[event_index]

        prediction["verification_status"] = "HIT"
        prediction["matched_cyclone_sid"] = str(event["SID"])
        prediction["matched_cyclone_name"] = str(event["cyclone_name"])
        prediction["genesis_time_utc"] = pd.Timestamp(
            event["genesis_time"]
        ).isoformat()
        prediction["actual_lead_hours"] = float(
            event["actual_lead_hours"]
        )
        prediction["genesis_latitude"] = float(
            event["genesis_latitude"]
        )
        prediction["genesis_longitude"] = normalize_longitude_180(
            float(event["genesis_longitude"])
        )
        prediction["genesis_distance_km"] = float(distance)
        prediction["lead_error_hours"] = float(
            event["lead_error_hours"]
        )

        used_predictions.add(prediction_index)
        used_events.add(event_index)

    for index, prediction in enumerate(annotated):
        if index not in used_predictions:
            prediction["verification_status"] = "FALSE_ALARM"
            prediction["matched_cyclone_sid"] = ""
            prediction["matched_cyclone_name"] = ""
            prediction["genesis_time_utc"] = ""
            prediction["actual_lead_hours"] = np.nan
            prediction["genesis_latitude"] = np.nan
            prediction["genesis_longitude"] = np.nan
            prediction["genesis_distance_km"] = np.nan
            prediction["lead_error_hours"] = np.nan

    annotated_events: List[Dict[str, Any]] = []

    for event_index, event in enumerate(event_records):
        record = {
            "cyclone_sid": str(event["SID"]),
            "cyclone_name": str(event["cyclone_name"]),
            "genesis_time_utc": pd.Timestamp(
                event["genesis_time"]
            ).isoformat(),
            "genesis_latitude": float(event["genesis_latitude"]),
            "genesis_longitude": normalize_longitude_180(
                float(event["genesis_longitude"])
            ),
            "actual_lead_hours": float(
                event["actual_lead_hours"]
            ),
            "lead_error_hours": float(
                event["lead_error_hours"]
            ),
            "verification_status": (
                "HIT" if event_index in used_events else "MISS"
            ),
        }
        for optional in ("BASIN", "SUBBASIN", "SEASON"):
            if optional in event:
                record[optional.lower()] = event[optional]
        annotated_events.append(record)

    return annotated, annotated_events


# =============================================================================
# SINGLE-FILE EVALUATION
# =============================================================================

def evaluate_file(
    grib_path: Path,
    joint_model: tf.keras.Model,
    means: np.ndarray,
    stds: np.ndarray,
    genesis_catalog: pd.DataFrame,
    config: Config,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    analysis_time = parse_analysis_time(grib_path)
    valid_time = analysis_time + timedelta(hours=config.leadtime)

    logging.info(
        "Processing %s | analysis=%s | valid=%s",
        grib_path.name,
        analysis_time.isoformat(),
        valid_time.isoformat(),
    )

    ds = preprocess_grib(grib_path)
    full_grid = construct_grid(ds, means, stds)

    padded_grid = np.pad(
        full_grid,
        ((0, 0), (HALF_PATCH, HALF_PATCH), (0, 0)),
        mode="wrap",
    )

    indices, diagnostics = candidate_indices(ds, config)
    logging.info(
        "%s: %d physically plausible candidate grid cells",
        grib_path.name,
        len(indices),
    )

    raw: List[Dict[str, Any]] = []

    for start in tqdm(
        range(0, len(indices), config.batch_size),
        desc=f"Inference {grib_path.stem}",
        leave=False,
    ):
        batch_indices = indices[start:start + config.batch_size]
        x_batch = extract_patch_batch(padded_grid, batch_indices)

        ensemble_p, cnn_p, vit_p = predict_joint(
            joint_model,
            x_batch,
        )

        keep = (
            (ensemble_p >= config.threshold)
            & (cnn_p >= config.min_expert_confidence)
            & (vit_p >= config.min_expert_confidence)
        )

        for local_index in np.where(keep)[0]:
            lat_index = int(batch_indices[local_index, 0])
            lon_index = int(batch_indices[local_index, 1])

            longitude_180 = normalize_longitude_180(
                float(diagnostics["lon"][lon_index])
            )

            raw.append(
                {
                    "file": grib_path.stem,
                    "analysis_time_utc": analysis_time.isoformat(),
                    "valid_time_utc": valid_time.isoformat(),
                    "lead_time_hrs": config.leadtime,
                    "latitude": float(diagnostics["lat"][lat_index]),
                    "longitude": longitude_180,
                    "confidence": float(ensemble_p[local_index]),
                    "confidence_percent": round(
                        float(ensemble_p[local_index]) * 100.0,
                        2,
                    ),
                    "cnn_confidence": float(cnn_p[local_index]),
                    "vit_confidence": float(vit_p[local_index]),
                    "sst_k": float(
                        diagnostics["sst"][lat_index, lon_index]
                    ),
                    "midlevel_rh_percent": float(
                        diagnostics["rh_mid"][lat_index, lon_index]
                    ),
                    "vorticity_850_s1": float(
                        diagnostics["vort_850"][lat_index, lon_index]
                    ),
                    "pressure_anomaly_pa": float(
                        diagnostics["pressure_anomaly"][
                            lat_index, lon_index
                        ]
                    ),
                    "shear_200_850_ms": float(
                        diagnostics["shear_200_850"][
                            lat_index, lon_index
                        ]
                    ),
                    "lowlevel_divergence_s1": float(
                        diagnostics["lowlevel_divergence"][
                            lat_index, lon_index
                        ]
                    ),
                    "land_fraction": float(
                        diagnostics["land"][lat_index, lon_index]
                    ),
                }
            )

    clustered = cluster_detections(raw, config)

    events = candidate_events_for_analysis_time(
        genesis_catalog,
        analysis_time,
        target_lead_hours=float(config.leadtime),
        lead_tolerance_hours=config.lead_tolerance_hours,
    )

    annotated_predictions, annotated_events = match_predictions_to_events(
        clustered,
        events,
        config,
    )

    hits = sum(
        item["verification_status"] == "HIT"
        for item in annotated_predictions
    )
    false_alarms = sum(
        item["verification_status"] == "FALSE_ALARM"
        for item in annotated_predictions
    )
    misses = sum(
        item["verification_status"] == "MISS"
        for item in annotated_events
    )

    summary = {
        "file": grib_path.stem,
        "analysis_time_utc": analysis_time.isoformat(),
        "valid_time_utc": valid_time.isoformat(),
        "raw_neural_detections": len(raw),
        "final_disturbances": len(annotated_predictions),
        "observed_genesis_events": len(annotated_events),
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "cyclones_occurring": "; ".join(
            sorted({
                str(item["cyclone_name"])
                for item in annotated_events
            })
        ),
        "cyclones_hit": "; ".join(
            sorted({
                str(item["matched_cyclone_name"])
                for item in annotated_predictions
                if item["verification_status"] == "HIT"
            })
        ),
    }

    return annotated_predictions, annotated_events, summary


# =============================================================================
# EXPORT
# =============================================================================

def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def export_geojson(
    predictions: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    features = []

    for prediction in predictions:
        properties = {
            key: json_safe(value)
            for key, value in prediction.items()
            if key not in {"latitude", "longitude"}
        }

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        float(prediction["longitude"]),
                        float(prediction["latitude"]),
                    ],
                },
                "properties": properties,
            }
        )

    document = {
        "type": "FeatureCollection",
        "features": features,
    }

    path.write_text(
        json.dumps(document, indent=2),
        encoding="utf-8",
    )


def aggregate_unique_storm_metrics(
    event_frame: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Collapse repeated forecast cycles by cyclone SID.

    A unique cyclone is counted as detected when at least one valid pre-genesis
    forecast cycle matched it within the configured spatial tolerance.
    """
    if event_frame.empty or "cyclone_sid" not in event_frame.columns:
        return {
            "unique_storms": 0,
            "unique_storm_hits": 0,
            "unique_storm_misses": 0,
            "unique_storm_pod": None,
            "unique_storm_names_hit": [],
            "unique_storm_names_missed": [],
        }

    grouped = event_frame.groupby("cyclone_sid", dropna=False)

    rows = []
    for sid, group in grouped:
        hit = bool((group["verification_status"] == "HIT").any())
        name = str(group["cyclone_name"].iloc[0])
        rows.append(
            {
                "cyclone_sid": sid,
                "cyclone_name": name,
                "hit": hit,
            }
        )

    unique = pd.DataFrame(rows)
    hits = int(unique["hit"].sum())
    total = int(len(unique))
    misses = total - hits

    return {
        "unique_storms": total,
        "unique_storm_hits": hits,
        "unique_storm_misses": misses,
        "unique_storm_pod": hits / total if total else None,
        "unique_storm_names_hit": sorted(
            unique.loc[unique["hit"], "cyclone_name"].astype(str).tolist()
        ),
        "unique_storm_names_missed": sorted(
            unique.loc[~unique["hit"], "cyclone_name"].astype(str).tolist()
        ),
    }


def aggregate_metrics(
    prediction_frame: pd.DataFrame,
    event_frame: pd.DataFrame,
) -> Dict[str, Any]:
    hits = int(
        (prediction_frame.get("verification_status", pd.Series(dtype=str)) == "HIT").sum()
    )
    false_alarms = int(
        (
            prediction_frame.get(
                "verification_status",
                pd.Series(dtype=str),
            )
            == "FALSE_ALARM"
        ).sum()
    )
    misses = int(
        (event_frame.get("verification_status", pd.Series(dtype=str)) == "MISS").sum()
    )

    observed = hits + misses
    predicted = hits + false_alarms

    pod = hits / observed if observed else np.nan
    precision = hits / predicted if predicted else np.nan
    far = false_alarms / predicted if predicted else np.nan
    csi = hits / (hits + misses + false_alarms) if (
        hits + misses + false_alarms
    ) else np.nan

    hit_distances = (
        prediction_frame.loc[
            prediction_frame.get(
                "verification_status",
                pd.Series(index=prediction_frame.index, dtype=str),
            )
            == "HIT",
            "genesis_distance_km",
        ]
        if "genesis_distance_km" in prediction_frame.columns
        else pd.Series(dtype=float)
    )

    cyclone_names = sorted(
        set(
            event_frame.get(
                "cyclone_name",
                pd.Series(dtype=str),
            )
            .dropna()
            .astype(str)
        )
    )

    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "probability_of_detection": json_safe(pod),
        "precision": json_safe(precision),
        "false_alarm_ratio": json_safe(far),
        "critical_success_index": json_safe(csi),
        "mean_hit_distance_km": json_safe(hit_distances.mean()),
        "median_hit_distance_km": json_safe(hit_distances.median()),
        "number_of_observed_cyclones": len(cyclone_names),
        "cyclones_occurring": cyclone_names,
    }


# =============================================================================
# MAIN
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download native NCEP FNL analyses exactly one model lead time "
            "before observed IBTrACS genesis and verify the trained model."
        )
    )

    parser.add_argument(
        "--leadtime",
        type=int,
        default=24,
        choices=(24, 48, 72),
    )
    parser.add_argument(
        "--ibtracs-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--genesis-definition",
        choices=("first-track", "first-tropical"),
        default="first-track",
    )
    parser.add_argument(
        "--start-date",
        default="2026-01-01",
        help="Inclusive genesis date, YYYY-MM-DD. Default: 2026-01-01",
    )
    parser.add_argument(
        "--end-date",
        default="2027-01-01",
        help="Exclusive genesis date, YYYY-MM-DD. Default: 2027-01-01",
    )
    parser.add_argument(
        "--fnl-cache-dir",
        type=Path,
        default=FNL_CACHE_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "fnl_realworld_verification",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download required FNL files and exit without model inference.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=4,
        help=(
            "Number of simultaneous FNL downloads. Network/proxy bandwidth, "
            "not CPU cores, is usually the limiting factor. Default: 4."
        ),
    )
    parser.add_argument(
        "--insecure-ssl",
        action="store_true",
        help="Disable TLS verification only when an institutional proxy requires it.",
    )

    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=0.82,
    )
    parser.add_argument(
        "--min-expert-confidence",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument("--min-latitude", type=float, default=-35.0)
    parser.add_argument("--max-latitude", type=float, default=35.0)
    parser.add_argument(
        "--maximum-land-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--minimum-sst-k",
        type=float,
        default=299.15,
    )
    parser.add_argument(
        "--minimum-midlevel-rh-percent",
        type=float,
        default=55.0,
    )
    parser.add_argument(
        "--minimum-vorticity-850-s1",
        type=float,
        default=1.0e-5,
    )
    parser.add_argument(
        "--maximum-pressure-anomaly-pa",
        type=float,
        default=-100.0,
    )
    parser.add_argument(
        "--maximum-shear-200-850-ms",
        type=float,
        default=15.0,
        help="Maximum 200-850 hPa vector wind shear in m/s.",
    )
    parser.add_argument(
        "--maximum-lowlevel-divergence-s1",
        type=float,
        default=-1.0e-6,
        help=(
            "Maximum accepted 850-hPa divergence. Negative values require "
            "low-level convergence."
        ),
    )

    parser.add_argument(
        "--cluster-radius-km",
        type=float,
        default=600.0,
    )
    parser.add_argument(
        "--min-cluster-support",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--lead-tolerance-hours",
        type=float,
        default=0.0,
        help=(
            "Lead-time tolerance around the exact target. Default 0 means "
            "strict genesis exactly at analysis + model lead time."
        ),
    )
    parser.add_argument(
        "--ibtracs-distance-tolerance-km",
        type=float,
        default=500.0,
    )

    return parser


def validate_args(args: argparse.Namespace) -> Tuple[datetime, datetime]:
    if not 0.0 < args.alert_threshold < 1.0:
        raise ValueError("--alert-threshold must be between 0 and 1")
    if not 0.0 <= args.min_expert_confidence < 1.0:
        raise ValueError("--min-expert-confidence must be in [0,1)")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.cluster_radius_km <= 0:
        raise ValueError("--cluster-radius-km must be positive")
    if args.maximum_shear_200_850_ms <= 0:
        raise ValueError("--maximum-shear-200-850-ms must be positive")
    if args.min_cluster_support < 1:
        raise ValueError("--min-cluster-support must be at least 1")
    if args.max_detections < 1:
        raise ValueError("--max-detections must be at least 1")
    if args.lead_tolerance_hours < 0:
        raise ValueError("--lead-tolerance-hours cannot be negative")
    if args.download_workers < 1:
        raise ValueError("--download-workers must be at least 1")
    if args.download_workers > 12:
        raise ValueError(
            "--download-workers above 12 is not recommended for the archive/proxy"
        )

    start_date = datetime.fromisoformat(args.start_date).replace(
        tzinfo=timezone.utc
    )
    end_date = datetime.fromisoformat(args.end_date).replace(
        tzinfo=timezone.utc
    )
    if end_date <= start_date:
        raise ValueError("--end-date must be later than --start-date")

    return start_date, end_date


def main() -> None:
    args = build_parser().parse_args()
    start_date, end_date = validate_args(args)

    config = Config(
        leadtime=args.leadtime,
        threshold=args.alert_threshold,
        min_expert_confidence=args.min_expert_confidence,
        batch_size=args.batch_size,
        min_latitude=args.min_latitude,
        max_latitude=args.max_latitude,
        maximum_land_fraction=args.maximum_land_fraction,
        minimum_sst_k=args.minimum_sst_k,
        minimum_midlevel_rh_percent=args.minimum_midlevel_rh_percent,
        minimum_vorticity_850_s1=args.minimum_vorticity_850_s1,
        maximum_pressure_anomaly_pa=args.maximum_pressure_anomaly_pa,
        maximum_shear_200_850_ms=args.maximum_shear_200_850_ms,
        maximum_lowlevel_divergence_s1=(
            args.maximum_lowlevel_divergence_s1
        ),
        cluster_radius_km=args.cluster_radius_km,
        min_cluster_support=args.min_cluster_support,
        max_detections=args.max_detections,
        lead_tolerance_hours=args.lead_tolerance_hours,
        ibtracs_distance_tolerance_km=(
            args.ibtracs_distance_tolerance_km
        ),
        genesis_definition=args.genesis_definition,
    )

    output_dir = args.output_dir.resolve()
    cache_dir = args.fnl_cache_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ibtracs = read_ibtracs(args.ibtracs_csv.resolve())
    genesis_catalog = build_genesis_catalog(
        ibtracs,
        args.genesis_definition,
    )

    period_events = genesis_catalog[
        (genesis_catalog["genesis_time"] >= pd.Timestamp(start_date))
        & (genesis_catalog["genesis_time"] < pd.Timestamp(end_date))
    ].copy()

    if period_events.empty:
        raise ValueError(
            "No IBTrACS genesis events were found in the requested period."
        )

    analysis_times = select_fnl_analysis_times(
        genesis_catalog,
        start_date=start_date,
        end_date=end_date,
        leadtime_hours=args.leadtime,
    )

    logging.info(
        "Found %d observed genesis events and %d unique required FNL cycles.",
        len(period_events),
        len(analysis_times),
    )

    downloaded_files: List[Path] = []
    download_failures: List[Dict[str, str]] = []

    logging.info(
        "Downloading %d FNL files with %d parallel workers.",
        len(analysis_times),
        args.download_workers,
    )

    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        future_to_time = {
            executor.submit(
                download_fnl_file,
                analysis_time,
                cache_dir,
                insecure_ssl=args.insecure_ssl,
            ): analysis_time
            for analysis_time in analysis_times
        }

        for future in as_completed(future_to_time):
            analysis_time = future_to_time[future]
            try:
                downloaded_files.append(future.result())
                logging.info(
                    "FNL download progress: %d/%d complete",
                    len(downloaded_files),
                    len(analysis_times),
                )
            except Exception as exc:
                logging.exception(
                    "Could not obtain FNL %s",
                    analysis_time.isoformat(),
                )
                download_failures.append(
                    {
                        "analysis_time_utc": analysis_time.isoformat(),
                        "url": fnl_download_url(analysis_time),
                        "error": str(exc),
                    }
                )

    downloaded_files = sorted(downloaded_files)

    manifest_path = output_dir / "fnl_download_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "requested_start_date": start_date.isoformat(),
                "requested_end_date": end_date.isoformat(),
                "lead_time_hours": args.leadtime,
                "requested_cycles": len(analysis_times),
                "downloaded_cycles": len(downloaded_files),
                "download_failures": download_failures,
                "files": [str(path) for path in downloaded_files],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.download_only:
        print(f"Downloaded {len(downloaded_files)} FNL files.")
        print(f"Manifest: {manifest_path}")
        return

    if not downloaded_files:
        raise RuntimeError("No FNL files are available for evaluation.")

    means, stds = load_normalization()
    _, _, _, joint_model = load_models(args.leadtime)

    all_predictions: List[Dict[str, Any]] = []
    all_events: List[Dict[str, Any]] = []
    all_file_summaries: List[Dict[str, Any]] = []
    processing_failures: List[Dict[str, str]] = []

    for path in downloaded_files:
        try:
            predictions, events, summary = evaluate_file(
                path,
                joint_model,
                means,
                stds,
                genesis_catalog,
                config,
            )

            all_predictions.extend(predictions)

            for event in events:
                event["file"] = path.stem
                event["analysis_time_utc"] = summary["analysis_time_utc"]
                event["valid_time_utc"] = summary["valid_time_utc"]
                event["lead_time_hrs"] = args.leadtime
                all_events.append(event)

            all_file_summaries.append(summary)

            logging.info(
                "%s: cyclones=%s | hits=%d misses=%d false alarms=%d",
                path.name,
                summary["cyclones_occurring"] or "NONE",
                summary["hits"],
                summary["misses"],
                summary["false_alarms"],
            )

        except Exception as exc:
            logging.exception("Failed %s", path.name)
            processing_failures.append(
                {
                    "file": path.name,
                    "error": str(exc),
                }
            )

    prediction_frame = pd.DataFrame(all_predictions)
    event_frame = pd.DataFrame(all_events)
    file_summary_frame = pd.DataFrame(all_file_summaries)

    stem = f"fnl_2026_verified_{args.leadtime}h"
    prediction_csv = output_dir / f"{stem}_predictions.csv"
    event_csv = output_dir / f"{stem}_cyclones.csv"
    file_summary_csv = output_dir / f"{stem}_file_summary.csv"
    geojson_path = output_dir / f"{stem}.geojson"
    metrics_path = output_dir / f"{stem}_metrics.json"
    failures_path = output_dir / f"{stem}_failures.json"

    prediction_frame.to_csv(prediction_csv, index=False)
    event_frame.to_csv(event_csv, index=False)
    file_summary_frame.to_csv(file_summary_csv, index=False)
    export_geojson(all_predictions, geojson_path)

    metrics = aggregate_metrics(prediction_frame, event_frame)
    metrics.update(aggregate_unique_storm_metrics(event_frame))
    metrics.update(
        {
            "data_source": "NCEP FNL d083002 native 1-degree GRIB2",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "lead_time_hours": args.leadtime,
            "lead_tolerance_hours": args.lead_tolerance_hours,
            "alert_threshold": args.alert_threshold,
            "minimum_expert_confidence": args.min_expert_confidence,
            "maximum_shear_200_850_ms": args.maximum_shear_200_850_ms,
            "maximum_lowlevel_divergence_s1": (
                args.maximum_lowlevel_divergence_s1
            ),
            "cluster_radius_km": args.cluster_radius_km,
            "minimum_cluster_support": args.min_cluster_support,
            "ibtracs_distance_tolerance_km": (
                args.ibtracs_distance_tolerance_km
            ),
            "genesis_definition": args.genesis_definition,
            "requested_fnl_cycles": len(analysis_times),
            "downloaded_fnl_cycles": len(downloaded_files),
            "processed_files": len(all_file_summaries),
            "download_failures": len(download_failures),
            "processing_failures": len(processing_failures),
        }
    )

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    failures_path.write_text(
        json.dumps(
            {
                "download_failures": download_failures,
                "processing_failures": processing_failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    print("NCEP FNL REAL-WORLD TROPICAL CYCLOGENESIS VERIFICATION")
    print("=" * 76)
    print(f"Requested FNL cycles : {len(analysis_times)}")
    print(f"Downloaded cycles    : {len(downloaded_files)}")
    print(f"Processed files      : {metrics['processed_files']}")
    print(f"Cycle hits           : {metrics['hits']}")
    print(f"Cycle misses         : {metrics['misses']}")
    print(f"False alarms         : {metrics['false_alarms']}")
    print(f"Precision            : {metrics['precision']}")
    print(f"POD                  : {metrics['probability_of_detection']}")
    print(f"FAR                  : {metrics['false_alarm_ratio']}")
    print(f"CSI                  : {metrics['critical_success_index']}")
    print(f"Unique storms        : {metrics['unique_storms']}")
    print(f"Unique hits          : {metrics['unique_storm_hits']}")
    print(f"Unique misses        : {metrics['unique_storm_misses']}")
    print(f"Unique-storm POD     : {metrics['unique_storm_pod']}")
    print("Cyclones detected:")
    for name in metrics["unique_storm_names_hit"]:
        print(f"  - {name}")
    print("Cyclones missed:")
    for name in metrics["unique_storm_names_missed"]:
        print(f"  - {name}")
    print()
    print(f"Predictions CSV      : {prediction_csv}")
    print(f"Cyclones CSV         : {event_csv}")
    print(f"File summary         : {file_summary_csv}")
    print(f"GeoJSON              : {geojson_path}")
    print(f"Metrics JSON         : {metrics_path}")
    print(f"Download manifest    : {manifest_path}")


if __name__ == "__main__":
    main()
