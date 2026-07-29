#!/usr/bin/env python3
"""
Operational Tropical Cyclogenesis Inference Pipeline
====================================================

Scientifically aligned with the project pipeline:

- 137 channels:
    7 pressure-level variables × 19 pressure levels = 133
    4 surface variables = 4
- 21 × 21 spatial patches
- Chronology-specific models for 24 h, 48 h, and 72 h lead times
- Exact channel-schema validation
- Training-only normalization statistics
- Per-horizon validation thresholds
- No cross-horizon model fallback
- Robust GRIB variable extraction and unit harmonization
- Great-circle non-maximum suppression
- Mean channel attribution for fair variable comparison
- Explicit warning/error for NCAR-training to GDAS/GFS operational domain shift

Important
---------
The historical project was trained on NCEP/NCAR-family atmospheric data.
This script downloads operational GDAS/GFS fields. That is a domain shift.
Use --allow-domain-shift only after validating model behaviour on overlapping
historical GDAS/GFS data or after bias correction/fine-tuning.

Required files in MODELS_DIR
----------------------------
best_cnn_lstm_24h.keras
best_vit_gru_24h.keras
best_ensemble_24h.keras

best_cnn_lstm_48h.keras
best_vit_gru_48h.keras
best_ensemble_48h.keras

best_cnn_lstm_72h.keras
best_vit_gru_72h.keras
best_ensemble_72h.keras

channel_means.npy
channel_stds.npy
channel_names.json
operational_thresholds.json
training_metadata.json

Example operational_thresholds.json
-----------------------------------
{
  "24": 0.43,
  "48": 0.39,
  "72": 0.35
}

Example training_metadata.json
------------------------------
{
  "dataset": "NCEP_NCAR",
  "grid_resolution_deg": 1.0,
  "patch_size": 21,
  "num_channels": 137
}
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import requests
import tensorflow as tf
import urllib3
import xarray as xr
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


# =============================================================================
# PATHS AND PROJECT IMPORTS
# =============================================================================

BASE_DIR = Path(os.getenv("TC_BASE_DIR", "/workspace")).resolve()
SCRIPTS_DIR = BASE_DIR / "scripts"
LIVE_DATA_DIR = BASE_DIR / "live_data"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "saved_models"

FNL_HTTP_ROOT = (
    "https://thredds.rda.ucar.edu/thredds/fileServer/"
    "files/g/d083002/grib2"
)

# Strict 2026 FNL retrospective verification using the selected balanced setup:
# 16 hits, 36 false alarms, 32 observed genesis events.
REAL_WORLD_PRECISION_REFERENCE = 16.0 / (16.0 + 36.0)
REAL_WORLD_POD_REFERENCE = 16.0 / 32.0
REAL_WORLD_CSI_REFERENCE = 16.0 / (16.0 + 16.0 + 36.0)

for directory in (LIVE_DATA_DIR, OUTPUT_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

for path in (str(SCRIPTS_DIR), str(BASE_DIR)):
    if path not in sys.path:
        sys.path.append(path)

try:
    from train_ensemble import build_advanced_ensemble
    from train_ensemble import LayerScale
    from train_vit_gru import PatchExtractorAndEmbedder, SqueezeAndExcitationBlock
except ImportError as exc:
    raise ImportError(
        "Unable to import the custom model builders/layers. "
        "Ensure /workspace/scripts contains train_ensemble.py and "
        "train_vit_gru.py and that their dependencies are installed."
    ) from exc


# =============================================================================
# LOGGING AND TENSORFLOW
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    stream=sys.stdout,
)

tf.config.optimizer.set_experimental_options({"layout_optimizer": False})


# =============================================================================
# SCIENTIFIC CHANNEL SCHEMA
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
    1000.0,
    975.0,
    950.0,
    925.0,
    900.0,
    850.0,
    800.0,
    750.0,
    700.0,
    650.0,
    600.0,
    550.0,
    500.0,
    450.0,
    400.0,
    350.0,
    300.0,
    250.0,
    200.0,
)

PATCH_SIZE = 21
HALF_PATCH = PATCH_SIZE // 2
EXPECTED_CHANNELS = len(V3D_NAMES) * len(PRESSURE_LEVELS) + len(V2D_NAMES)

if EXPECTED_CHANNELS != 137:
    raise RuntimeError(f"Internal schema error: expected 137 channels, got {EXPECTED_CHANNELS}")


def build_channel_schema() -> List[str]:
    """
    Returns the exact variable-major channel ordering used by this script:

    ugrdprs_1000 ... ugrdprs_200,
    vgrdprs_1000 ... vgrdprs_200,
    ...
    absvprs_1000 ... absvprs_200,
    pressfc, capesfc, tmpsfc, landmask
    """
    channels: List[str] = []
    for variable in V3D_NAMES:
        for level in PRESSURE_LEVELS:
            channels.append(f"{variable}_{int(level)}")
    channels.extend(V2D_NAMES)
    return channels


CHANNEL_SCHEMA = build_channel_schema()


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class RuntimeConfig:
    allow_domain_shift: bool
    insecure_ssl: bool
    threshold_file: Path
    alert_threshold: float
    training_metadata_file: Path
    channel_names_file: Path
    means_file: Path
    stds_file: Path
    batch_size: int
    nms_radius_km: float
    min_expert_confidence: float
    max_detections: int
    active_systems_json: Optional[Path]
    classification_mode: str
    post_classifier_model: Optional[Path]
    stale_track_radius_km: float
    same_system_radius_km: float
    min_latitude: float
    max_latitude: float
    maximum_land_fraction: float
    minimum_surface_temperature_k: Optional[float]
    poll_interval_hours: int
    lead_times: Tuple[int, ...]


# =============================================================================
# NETWORK SESSION
# =============================================================================

def get_requests_session(insecure_ssl: bool = False) -> requests.Session:
    """
    Creates a retry-enabled HTTP session.

    SSL verification remains enabled by default.
    Use --insecure-ssl only when the institutional proxy cannot be configured
    with a trusted CA certificate.
    """
    session = requests.Session()
    retries = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    adapter = HTTPAdapter(
        max_retries=retries,
        pool_connections=10,
        pool_maxsize=20,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.verify = not insecure_ssl
    if insecure_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        logging.warning(
            "SSL certificate verification is disabled. "
            "Install the institutional proxy CA certificate when possible."
        )

    http_proxy = os.getenv("http_proxy") or os.getenv("HTTP_PROXY")
    https_proxy = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
    if http_proxy or https_proxy:
        session.proxies = {
            "http": http_proxy,
            "https": https_proxy or http_proxy,
        }

    session.headers.update(
        {
            "User-Agent": (
                "TC-Genesis-Research/1.0 "
                "(operational inference; scientific research use)"
            )
        }
    )
    return session


# =============================================================================
# GRIB EXTRACTION
# =============================================================================

def load_grib_dataset(
    path: Path,
    *,
    short_name: str,
    type_of_level: str,
    extra_keys: Optional[Mapping[str, Any]] = None,
) -> xr.Dataset:
    filter_by_keys: Dict[str, Any] = {
        "shortName": short_name,
        "typeOfLevel": type_of_level,
    }
    if extra_keys:
        filter_by_keys.update(extra_keys)

    return xr.load_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "errors": "raise",
            "indexpath": "",
            "filter_by_keys": filter_by_keys,
        },
    )


def select_single_data_variable(ds: xr.Dataset, expected_short_name: str) -> xr.DataArray:
    """
    Returns the only meteorological data variable in a filtered GRIB dataset.
    Coordinates such as time, step and valid_time are ignored.
    """
    variables = list(ds.data_vars)
    if len(variables) != 1:
        raise ValueError(
            f"Expected one data variable for shortName={expected_short_name}, "
            f"found {variables}"
        )
    return ds[variables[0]]


def normalize_coordinate_names(da: xr.DataArray) -> xr.DataArray:
    rename_map: Dict[str, str] = {}
    for old, new in (
        ("latitude", "lat"),
        ("longitude", "lon"),
        ("isobaricInhPa", "lev"),
    ):
        if old in da.dims or old in da.coords:
            rename_map[old] = new

    if rename_map:
        da = da.rename(rename_map)

    required = {"lat", "lon"}
    missing = required - set(da.dims) - set(da.coords)
    if missing:
        raise ValueError(f"Missing geographic coordinates: {sorted(missing)}")

    return da


def drop_scalar_nonspatial_coordinates(da: xr.DataArray) -> xr.DataArray:
    """
    Removes scalar GRIB metadata coordinates that can conflict during merge.
    """
    removable = [
        name
        for name, coord in da.coords.items()
        if coord.ndim == 0 and name not in {"lat", "lon", "lev"}
    ]
    if removable:
        da = da.drop_vars(removable)
    return da


def standardize_lat_lon(da: xr.DataArray) -> xr.DataArray:
    da = normalize_coordinate_names(da)
    da = drop_scalar_nonspatial_coordinates(da)

    # Latitude ascending: -90 to 90.
    if da["lat"].values[0] > da["lat"].values[-1]:
        da = da.sortby("lat")

    # Longitude transformed to 0..360 and sorted.
    lon = np.mod(da["lon"].values.astype(np.float64), 360.0)
    da = da.assign_coords(lon=lon).sortby("lon")

    # Remove duplicated longitude values if present.
    unique_lon, unique_indices = np.unique(da["lon"].values, return_index=True)
    if len(unique_lon) != da.sizes["lon"]:
        da = da.isel(lon=np.sort(unique_indices))

    return da


def harmonize_units(variable: str, da: xr.DataArray) -> xr.DataArray:
    """
    Converts common operational GRIB representations into the canonical units
    expected by the project.

    Canonical units:
    - wind: m s-1
    - vertical velocity: Pa s-1
    - temperature: K
    - relative humidity: %
    - geopotential height: m
    - absolute vorticity: s-1
    - surface pressure: Pa
    - CAPE: J kg-1
    - land mask: 0..1
    """
    units = str(da.attrs.get("units", "")).strip().lower()
    values = da.astype(np.float32)

    if variable == "rhprs":
        finite_max = float(values.max(skipna=True).values)
        if units in {"1", "fraction", "proportion"} or finite_max <= 1.5:
            logging.info("Converting relative humidity from fraction to percent.")
            values = values * 100.0
        values.attrs["units"] = "%"

    elif variable == "pressfc":
        if units in {"hpa", "mb", "mbar", "millibar"}:
            logging.info("Converting surface pressure from hPa to Pa.")
            values = values * 100.0
        values.attrs["units"] = "Pa"

    elif variable == "hgtprs":
        if "m**2" in units or "m2 s-2" in units or "m^2 s^-2" in units:
            logging.info("Converting geopotential to geopotential height.")
            values = values / np.float32(9.80665)
        values.attrs["units"] = "m"

    elif variable == "landmask":
        values = values.clip(min=0.0, max=1.0)
        values.attrs["units"] = "1"

    return values


PRESSURE_GRIB_SPECS: Mapping[str, str] = {
    "ugrdprs": "u",
    "vgrdprs": "v",
    "vvelprs": "w",
    "tmpprs": "t",
    "rhprs": "r",
    "hgtprs": "gh",
    "absvprs": "absv",
}


# Several operational products encode CAPE or surface pressure differently.
SURFACE_GRIB_CANDIDATES: Mapping[str, Sequence[Mapping[str, Any]]] = {
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


def extract_pressure_variable(grib_path: Path, project_name: str) -> xr.DataArray:
    short_name = PRESSURE_GRIB_SPECS[project_name]
    ds = load_grib_dataset(
        grib_path,
        short_name=short_name,
        type_of_level="isobaricInhPa",
    )
    da = select_single_data_variable(ds, short_name)
    da = standardize_lat_lon(da)

    if "lev" not in da.dims and "lev" not in da.coords:
        raise ValueError(f"{project_name} has no pressure-level coordinate.")

    available_levels = {float(level) for level in da["lev"].values}
    missing_levels = [
        level for level in PRESSURE_LEVELS if float(level) not in available_levels
    ]
    if missing_levels:
        raise ValueError(
            f"{project_name} is missing required pressure levels: {missing_levels}"
        )

    # Select, rather than reindex, so missing fields cannot become silent NaNs.
    da = da.sel(lev=list(PRESSURE_LEVELS))
    da = harmonize_units(project_name, da)
    da.name = project_name
    return da


def extract_surface_variable(grib_path: Path, project_name: str) -> xr.DataArray:
    errors: List[str] = []

    for candidate in SURFACE_GRIB_CANDIDATES[project_name]:
        short_name = str(candidate["shortName"])
        type_of_level = str(candidate["typeOfLevel"])
        extra = {
            key: value
            for key, value in candidate.items()
            if key not in {"shortName", "typeOfLevel"}
        }

        try:
            ds = load_grib_dataset(
                grib_path,
                short_name=short_name,
                type_of_level=type_of_level,
                extra_keys=extra,
            )
            da = select_single_data_variable(ds, short_name)
            da = standardize_lat_lon(da)

            # Squeeze singleton non-spatial dimensions only.
            for dim in list(da.dims):
                if dim not in {"lat", "lon"}:
                    if da.sizes[dim] != 1:
                        raise ValueError(
                            f"{project_name} has unexpected dimension {dim}="
                            f"{da.sizes[dim]}"
                        )
                    da = da.squeeze(dim, drop=True)

            if set(da.dims) != {"lat", "lon"}:
                raise ValueError(
                    f"{project_name} must be a 2-D lat/lon field; got {da.dims}"
                )

            da = harmonize_units(project_name, da)
            da.name = project_name
            return da

        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    raise ValueError(
        f"Unable to extract required surface variable {project_name}. "
        f"Attempts: {' | '.join(errors)}"
    )


def validate_dataset(ds: xr.Dataset) -> None:
    required = set(V3D_NAMES) | set(V2D_NAMES)
    missing = required - set(ds.data_vars)
    if missing:
        raise ValueError(f"Dataset is missing required variables: {sorted(missing)}")

    for variable in V3D_NAMES:
        da = ds[variable]
        if tuple(da.dims) != ("lev", "lat", "lon"):
            # xarray may preserve a different dimension ordering.
            ds[variable] = da.transpose("lev", "lat", "lon")

        levels = [float(value) for value in ds[variable]["lev"].values]
        if levels != list(PRESSURE_LEVELS):
            raise ValueError(
                f"{variable} level order mismatch. "
                f"Expected {list(PRESSURE_LEVELS)}, got {levels}"
            )

    for variable in V2D_NAMES:
        da = ds[variable]
        if set(da.dims) != {"lat", "lon"}:
            raise ValueError(f"{variable} has invalid dimensions: {da.dims}")
        ds[variable] = da.transpose("lat", "lon")

    if not np.all(np.diff(ds["lat"].values) > 0):
        raise ValueError("Latitude coordinate must be strictly ascending.")

    if not np.all(np.diff(ds["lon"].values) > 0):
        raise ValueError("Longitude coordinate must be strictly ascending.")

    if float(ds["lon"].min()) < 0.0 or float(ds["lon"].max()) >= 360.0:
        raise ValueError("Longitude coordinate must use the [0, 360) convention.")


def preprocess_raw_grib(grib_path: Path) -> xr.Dataset:
    """
    Extracts the exact 137-channel meteorological schema as an xarray Dataset.

    Fails loudly when a required field or level is absent.
    """
    logging.info("Extracting and validating meteorological fields from %s", grib_path)

    arrays: List[xr.DataArray] = []

    for variable in V3D_NAMES:
        logging.info("Extracting pressure variable: %s", variable)
        arrays.append(extract_pressure_variable(grib_path, variable))

    for variable in V2D_NAMES:
        logging.info("Extracting surface variable: %s", variable)
        arrays.append(extract_surface_variable(grib_path, variable))

    ds = xr.merge(arrays, join="exact", compat="override")
    validate_dataset(ds)
    return ds


# =============================================================================
# OPERATIONAL GDAS/GFS DOWNLOAD
# =============================================================================

def fnl_archive_url(target_time: datetime) -> str:
    year = target_time.strftime("%Y")
    year_month = target_time.strftime("%Y.%m")
    filename = target_time.strftime("fnl_%Y%m%d_%H_00.grib2")
    return f"{FNL_HTTP_ROOT}/{year}/{year_month}/{filename}"


def download_fnl_analysis(
    target_time: datetime,
    *,
    session: requests.Session,
) -> Optional[Path]:
    """Download the native 1-degree NCEP FNL analysis used by training."""
    if target_time.tzinfo is None:
        target_time = target_time.replace(tzinfo=timezone.utc)

    filename = target_time.strftime("fnl_%Y%m%d_%H_00.grib2")
    local_path = LIVE_DATA_DIR / filename

    if local_path.exists() and local_path.stat().st_size > 10_000_000:
        logging.info("Using cached FNL analysis: %s", local_path)
        return local_path

    url = fnl_archive_url(target_time)
    temporary = local_path.with_suffix(".grib2.tmp")

    try:
        logging.info("Requesting NCEP FNL analysis: %s", url)
        with session.get(url, stream=True, timeout=(20, 240)) as response:
            if response.status_code == 404:
                return None
            response.raise_for_status()

            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

        if temporary.stat().st_size < 10_000_000:
            temporary.unlink(missing_ok=True)
            return None

        temporary.replace(local_path)
        logging.info(
            "Downloaded FNL %.1f MB",
            local_path.stat().st_size / 1e6,
        )
        return local_path
    except requests.RequestException as exc:
        temporary.unlink(missing_ok=True)
        logging.warning("FNL download failed for %s: %s", target_time, exc)
        return None


# =============================================================================
# TRAINING/INFERENCE CONSISTENCY
# =============================================================================

def load_json_file(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_channel_schema_file(path: Path) -> None:
    stored = load_json_file(path)
    if not isinstance(stored, list):
        raise ValueError(f"{path} must contain a JSON list of channel names.")

    if stored != CHANNEL_SCHEMA:
        mismatch_lines: List[str] = []
        for index, (expected, actual) in enumerate(
            zip(CHANNEL_SCHEMA, stored), start=0
        ):
            if expected != actual:
                mismatch_lines.append(
                    f"index {index}: expected={expected}, actual={actual}"
                )
                if len(mismatch_lines) >= 10:
                    break

        if len(stored) != len(CHANNEL_SCHEMA):
            mismatch_lines.append(
                f"length mismatch: expected {len(CHANNEL_SCHEMA)}, got {len(stored)}"
            )

        raise ValueError(
            "Operational channel order does not match the training schema. "
            + " | ".join(mismatch_lines)
        )


def load_normalization_statistics(
    means_path: Path,
    stds_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    if not means_path.exists() or not stds_path.exists():
        raise FileNotFoundError(
            f"Missing normalization files: {means_path}, {stds_path}"
        )

    means = np.asarray(np.load(means_path), dtype=np.float32).reshape(-1)
    stds = np.asarray(np.load(stds_path), dtype=np.float32).reshape(-1)

    # The training generator standardized channels 0..135 and left the
    # final landmask channel unchanged. Therefore saved statistics may contain
    # either 136 meteorological channels or the full 137-channel schema.
    if means.shape == (EXPECTED_CHANNELS - 1,):
        logging.info(
            "Loaded 136 meteorological means; appending landmask mean 0."
        )
        means = np.concatenate(
            [means, np.asarray([0.0], dtype=np.float32)]
        )
    if stds.shape == (EXPECTED_CHANNELS - 1,):
        logging.info(
            "Loaded 136 meteorological standard deviations; appending "
            "landmask standard deviation 1."
        )
        stds = np.concatenate(
            [stds, np.asarray([1.0], dtype=np.float32)]
        )

    if means.shape != (EXPECTED_CHANNELS,):
        raise ValueError(
            f"Expected 136 or {EXPECTED_CHANNELS} means, "
            f"received {means.shape}"
        )
    if stds.shape != (EXPECTED_CHANNELS,):
        raise ValueError(
            f"Expected 136 or {EXPECTED_CHANNELS} standard deviations, "
            f"received {stds.shape}"
        )

    if not np.all(np.isfinite(means)):
        raise ValueError("Normalization means contain non-finite values.")
    if not np.all(np.isfinite(stds)):
        raise ValueError("Normalization standard deviations contain non-finite values.")
    if np.any(stds <= 0):
        bad_indices = np.where(stds <= 0)[0].tolist()
        raise ValueError(
            f"Normalization standard deviations must be > 0. "
            f"Invalid channel indices: {bad_indices[:20]}"
        )

    return means, stds


def load_operational_thresholds(path: Path) -> Dict[int, float]:
    raw = load_json_file(path)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object.")

    thresholds: Dict[int, float] = {}
    for key, value in raw.items():
        leadtime = int(key)
        threshold = float(value)
        if not 0.0 < threshold < 1.0:
            raise ValueError(
                f"Threshold for {leadtime} h must be between 0 and 1."
            )
        thresholds[leadtime] = threshold

    return thresholds


def validate_training_metadata(
    path: Path,
    *,
    allow_domain_shift: bool,
) -> Dict[str, Any]:
    metadata = load_json_file(path)
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} must contain a JSON object.")

    dataset_name = str(metadata.get("dataset", "")).upper()
    num_channels = int(metadata.get("num_channels", -1))
    patch_size = int(metadata.get("patch_size", -1))

    if num_channels != EXPECTED_CHANNELS:
        raise ValueError(
            f"Training metadata expects {num_channels} channels; "
            f"FNL inference schema contains {EXPECTED_CHANNELS}."
        )

    if patch_size != PATCH_SIZE:
        raise ValueError(
            f"Training metadata expects patch size {patch_size}; "
            f"FNL inference uses {PATCH_SIZE}."
        )

    # Historical metadata sometimes used the broad name NCEP_NCAR even though
    # the extraction script and filenames show that the actual product was
    # NCEP FNL. Treat these names as compatible with native FNL input.
    fnl_family = {
        "NCEP_NCAR",
        "NCEP_FNL",
        "FNL",
        "NCEP-FNL",
        "GDAS_FNL",
    }

    if dataset_name not in fnl_family:
        message = (
            f"Training dataset is recorded as '{dataset_name}', while this "
            "script uses native NCEP FNL fields."
        )
        if allow_domain_shift:
            logging.warning(
                "%s Proceeding because domain-shift override is enabled.",
                message,
            )
        else:
            raise RuntimeError(message)
    else:
        logging.info(
            "Training source '%s' is treated as compatible with native "
            "NCEP FNL inference.",
            dataset_name,
        )

    return metadata


# =============================================================================
# MODEL LOADING
# =============================================================================

def model_checkpoint_paths(leadtime: int) -> Tuple[Path, Path, Path]:
    return (
        MODELS_DIR / f"best_cnn_lstm_{leadtime}h.keras",
        MODELS_DIR / f"best_vit_gru_{leadtime}h.keras",
        MODELS_DIR / f"best_ensemble_{leadtime}h.keras",
    )


def is_scalar_probability_model(model: tf.keras.Model) -> bool:
    shape = model.output_shape
    if isinstance(shape, list):
        return False
    return tuple(shape[1:]) == (1,)


def load_leadtime_models(
    leadtime: int,
) -> Tuple[tf.keras.Model, Optional[tf.keras.Model], Optional[tf.keras.Model]]:
    """
    Loads only the requested horizon.

    There is deliberately no fallback from 48 h/72 h to 24 h.
    """
    cnn_path, vit_path, ensemble_path = model_checkpoint_paths(leadtime)

    missing = [
        path
        for path in (cnn_path, vit_path, ensemble_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {leadtime} h checkpoint(s): "
            + ", ".join(str(path) for path in missing)
        )

    logging.info("Building %s h gated cross-attention ensemble.", leadtime)
    ensemble_model, cnn_expert, vit_expert = build_advanced_ensemble(
        str(cnn_path),
        str(vit_path),
        input_shape=(PATCH_SIZE, PATCH_SIZE, EXPECTED_CHANNELS),
    )
    ensemble_model.load_weights(str(ensemble_path))

    if not is_scalar_probability_model(ensemble_model):
        raise ValueError(
            f"Ensemble output must be shape (None, 1); got "
            f"{ensemble_model.output_shape}"
        )

    if cnn_expert is not None and not is_scalar_probability_model(cnn_expert):
        logging.warning(
            "CNN expert output is %s, not a scalar probability. "
            "cnn_confidence will be omitted.",
            cnn_expert.output_shape,
        )
        cnn_expert = None

    if vit_expert is not None and not is_scalar_probability_model(vit_expert):
        logging.warning(
            "ViT expert output is %s, not a scalar probability. "
            "vit_confidence will be omitted.",
            vit_expert.output_shape,
        )
        vit_expert = None

    return ensemble_model, cnn_expert, vit_expert


# =============================================================================
# GRID CONSTRUCTION
# =============================================================================

def interpolate_to_training_grid(ds: xr.Dataset) -> xr.Dataset:
    """
    Harmonizes operational 0.25-degree fields to the historical 1-degree grid.

    Interpolation does not remove domain shift; it only aligns geometry.
    """
    target_lats = np.arange(-90.0, 91.0, 1.0, dtype=np.float32)
    target_lons = np.arange(0.0, 360.0, 1.0, dtype=np.float32)

    interpolated = ds.interp(
        lat=target_lats,
        lon=target_lons,
        method="linear",
    )
    validate_dataset(interpolated)
    return interpolated


def construct_normalized_grid(
    ds: xr.Dataset,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    """
    Constructs the normalized [lat, lon, 137] tensor in the exact schema order.
    """
    layers: List[np.ndarray] = []
    channel_names_seen: List[str] = []
    index = 0

    for variable in V3D_NAMES:
        for level in PRESSURE_LEVELS:
            layer = np.asarray(
                ds[variable].sel(lev=level).values,
                dtype=np.float32,
            )

            if not np.all(np.isfinite(layer)):
                raise ValueError(
                    f"{variable}_{int(level)} contains missing/non-finite values."
                )

            normalized = (layer - means[index]) / stds[index]
            layers.append(normalized)
            channel_names_seen.append(f"{variable}_{int(level)}")
            index += 1

    for variable in V2D_NAMES:
        layer = np.asarray(ds[variable].values, dtype=np.float32)

        if not np.all(np.isfinite(layer)):
            raise ValueError(f"{variable} contains missing/non-finite values.")

        normalized = (layer - means[index]) / stds[index]
        layers.append(normalized)
        channel_names_seen.append(variable)
        index += 1

    if channel_names_seen != CHANNEL_SCHEMA:
        raise RuntimeError("Internal channel construction order mismatch.")

    grid = np.stack(layers, axis=-1).astype(np.float32)

    if grid.shape != (181, 360, EXPECTED_CHANNELS):
        raise ValueError(
            f"Expected normalized grid shape (181, 360, 137), got {grid.shape}"
        )

    if not np.all(np.isfinite(grid)):
        raise ValueError("Normalized grid contains non-finite values.")

    return grid


# =============================================================================
# XAI
# =============================================================================

def compute_gradient_attribution(
    model: tf.keras.Model,
    input_patch: np.ndarray,
) -> Tuple[str, Dict[str, float], Dict[str, float]]:
    """
    Computes absolute-gradient attribution.

    Fair variable ranking:
    - mean_attribution: averages the 19 pressure-level channels of each 3-D
      variable, allowing fair comparison with 2-D variables.
    - total_column_attribution: sums all levels of a 3-D variable and is kept
      separately because it measures total vertical-column contribution.

    This is gradient saliency, not Grad-CAM or Integrated Gradients.
    """
    x_tensor = tf.convert_to_tensor(input_patch.astype(np.float32))

    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        prediction = model(x_tensor, training=False)

    gradients = tape.gradient(prediction, x_tensor)
    if gradients is None:
        raise RuntimeError("Unable to compute input gradients.")

    per_channel = tf.reduce_mean(
        tf.abs(gradients),
        axis=(0, 1, 2),
    ).numpy()

    if per_channel.shape != (EXPECTED_CHANNELS,):
        raise ValueError(
            f"Expected {EXPECTED_CHANNELS} attribution channels, "
            f"got {per_channel.shape}"
        )

    mean_attribution: Dict[str, float] = {}
    total_column_attribution: Dict[str, float] = {}

    index = 0
    for variable in V3D_NAMES:
        values = per_channel[index : index + len(PRESSURE_LEVELS)]
        mean_attribution[variable] = float(np.mean(values))
        total_column_attribution[variable] = float(np.sum(values))
        index += len(PRESSURE_LEVELS)

    for variable in V2D_NAMES:
        value = float(per_channel[index])
        mean_attribution[variable] = value
        total_column_attribution[variable] = value
        index += 1

    total_mean = sum(mean_attribution.values())
    if total_mean > 0:
        mean_attribution = {
            key: value / total_mean
            for key, value in mean_attribution.items()
        }

    total_column = sum(total_column_attribution.values())
    if total_column > 0:
        total_column_attribution = {
            key: value / total_column
            for key, value in total_column_attribution.items()
        }

    top_variable = max(mean_attribution, key=mean_attribution.get)
    return top_variable, mean_attribution, total_column_attribution


# =============================================================================
# SPATIAL FILTERING
# =============================================================================

def haversine_distance_km(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    earth_radius_km = 6371.0088

    phi_1 = math.radians(latitude_1)
    phi_2 = math.radians(latitude_2)
    delta_phi = math.radians(latitude_2 - latitude_1)
    delta_lambda = math.radians(longitude_2 - longitude_1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi_1)
        * math.cos(phi_2)
        * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * earth_radius_km * math.asin(min(1.0, math.sqrt(a)))


def apply_spatial_nms(
    predictions: Sequence[Dict[str, Any]],
    distance_threshold_km: float,
) -> List[Dict[str, Any]]:
    """
    Great-circle non-maximum suppression.
    """
    remaining = sorted(
        predictions,
        key=lambda item: float(item["confidence"]),
        reverse=True,
    )
    retained: List[Dict[str, Any]] = []

    while remaining:
        best = remaining.pop(0)
        retained.append(best)

        filtered: List[Dict[str, Any]] = []
        for candidate in remaining:
            distance = haversine_distance_km(
                float(best["latitude"]),
                float(best["longitude"]),
                float(candidate["latitude"]),
                float(candidate["longitude"]),
            )
            if distance > distance_threshold_km:
                filtered.append(candidate)

        remaining = filtered

    return retained


# =============================================================================
# POST-CLASSIFICATION
# =============================================================================

ACTIVE_CYCLONE_STATUSES = {
    "TD", "TS", "STS", "TC", "TY", "HU", "CY", "ST", "SS", "SD"
}

INVEST_OR_DISTURBANCE_STATUSES = {
    "DB", "INVEST", "PTC", "LO", "WV"
}

DECAYING_SYSTEM_STATUSES = {
    "EX", "ET", "REMNANT", "RL", "DS"
}

GENERIC_SYSTEM_NAMES = {
    "", "UNKNOWN", "UNNAMED", "NONAME", "NO NAME",
    "TROPICAL", "DISTURBANCE", "TROPICAL DISTURBANCE"
}


def load_active_systems(path: Optional[Path]) -> List[Dict[str, Any]]:
    """
    Load optional active/recent storm positions.

    Accepted JSON forms:
      1. A list of storm dictionaries.
      2. {"systems": [...]}.

    Each item should contain:
      name, latitude, longitude, time, status

    Example:
      {
        "name": "DOLPHIN",
        "latitude": 12.5,
        "longitude": 133.0,
        "time": "2026-07-27T00:00:00+00:00",
        "status": "TS"
      }
    """
    if path is None:
        return []

    if not path.exists():
        raise FileNotFoundError(f"Active-systems JSON not found: {path}")

    payload = load_json_file(path)
    if isinstance(payload, dict):
        systems = payload.get("systems", [])
    elif isinstance(payload, list):
        systems = payload
    else:
        raise ValueError(
            "Active-systems JSON must be a list or an object containing "
            "a 'systems' list."
        )

    cleaned: List[Dict[str, Any]] = []
    for item in systems:
        if not isinstance(item, dict):
            continue

        required = {"latitude", "longitude", "time"}
        if not required.issubset(item):
            logging.warning(
                "Skipping active-system entry missing %s: %s",
                sorted(required - set(item)),
                item,
            )
            continue

        system_time = datetime.fromisoformat(
            str(item["time"]).replace("Z", "+00:00")
        )
        if system_time.tzinfo is None:
            system_time = system_time.replace(tzinfo=timezone.utc)

        storm_id = str(item.get("storm_id", "UNKNOWN")).upper()
        raw_name = str(item.get("name", "UNNAMED")).strip().upper()
        cleaned_name = (
            storm_id
            if raw_name in GENERIC_SYSTEM_NAMES and storm_id != "UNKNOWN"
            else raw_name
        )

        cleaned.append(
            {
                "name": cleaned_name,
                "storm_id": storm_id,
                "latitude": float(item["latitude"]),
                "longitude": float(item["longitude"]),
                "time": system_time,
                "status": str(item.get("status", "UNKNOWN")).upper(),
                "source": str(item.get("source", "UNKNOWN")),
                "basin": item.get("basin"),
            }
        )

    logging.info("Loaded %d active/recent system positions.", len(cleaned))
    return cleaned


def _cyclic_lon_slice(
    array: np.ndarray,
    lat_index: int,
    lon_index: int,
    radius: int,
) -> np.ndarray:
    """Return a latitude-bounded, longitude-cyclic neighbourhood."""
    lat_start = max(0, lat_index - radius)
    lat_end = min(array.shape[-2], lat_index + radius + 1)

    lon_indices = np.mod(
        np.arange(lon_index - radius, lon_index + radius + 1),
        array.shape[-1],
    )

    subset = np.take(array[..., lat_start:lat_end, :], lon_indices, axis=-1)
    return subset


def calculate_environmental_metrics(
    ds: xr.Dataset,
    *,
    lat_index: int,
    lon_index: int,
    radius: int = HALF_PATCH,
) -> Dict[str, float]:
    """
    Calculate simple physically interpretable fields around a candidate.

    These are diagnostics for post-classification, not a learned probability.
    """
    land = np.asarray(ds["landmask"].values, dtype=np.float32)
    pressure = np.asarray(ds["pressfc"].values, dtype=np.float32)

    land_patch = _cyclic_lon_slice(land, lat_index, lon_index, radius)
    pressure_patch = _cyclic_lon_slice(
        pressure, lat_index, lon_index, radius
    )

    center_pressure = float(pressure[lat_index, lon_index])
    pressure_anomaly = center_pressure - float(np.nanmean(pressure_patch))
    land_fraction = float(np.nanmean(land_patch))

    rh_levels = []
    for level in (700.0, 600.0, 500.0):
        if level in set(float(value) for value in ds["lev"].values):
            rh_levels.append(
                float(ds["rhprs"].sel(lev=level).values[lat_index, lon_index])
            )
    midlevel_rh = float(np.nanmean(rh_levels)) if rh_levels else float("nan")

    u200 = float(ds["ugrdprs"].sel(lev=200.0).values[lat_index, lon_index])
    v200 = float(ds["vgrdprs"].sel(lev=200.0).values[lat_index, lon_index])
    u850 = float(ds["ugrdprs"].sel(lev=850.0).values[lat_index, lon_index])
    v850 = float(ds["vgrdprs"].sel(lev=850.0).values[lat_index, lon_index])
    shear = float(math.hypot(u200 - u850, v200 - v850))

    absolute_vorticity = float(
        ds["absvprs"].sel(lev=850.0).values[lat_index, lon_index]
    )

    surface_temperature = float(ds["tmpsfc"].values[lat_index, lon_index])

    return {
        "land_fraction": land_fraction,
        "pressure_anomaly_pa": pressure_anomaly,
        "midlevel_rh_percent": midlevel_rh,
        "shear_200_850_ms": shear,
        "vorticity_850_s1": absolute_vorticity,
        "surface_temperature_k": surface_temperature,
    }



def load_post_classifier_model(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Load a safe JSON logistic post-classifier (no pickle execution)."""
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("model_type") != "logistic_regression":
        raise ValueError("Post-classifier JSON model_type must be logistic_regression.")
    required = {"feature_names", "coefficients", "intercept"}
    if not required.issubset(payload):
        raise ValueError(f"Post-classifier missing fields: {sorted(required-set(payload))}")
    return payload


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def build_post_classifier_features(
    detection: Mapping[str, Any],
    nearest_distance_km: float,
    nearest_age_hours: float,
    model: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Build the exact feature dictionary expected by the trained JSON model.

    The historical trainer writes XAI names as ``xai_<variable>`` while the
    operational GeoJSON exposes them as ``xai_mean_<variable>``.  This function
    supports both representations and uses the training medians only when a
    value is genuinely unavailable.
    """
    latitude = _safe_float(detection.get("latitude"))
    raw_vorticity = _safe_float(detection.get("vorticity_850_s1"))
    nested_xai = detection.get("xai_mean_attribution", {}) or {}

    features: Dict[str, float] = {
        "ensemble_score": _safe_float(
            detection.get("confidence", detection.get("raw_model_score"))
        ),
        "cnn_score": _safe_float(detection.get("cnn_confidence")),
        "vit_score": _safe_float(detection.get("vit_confidence")),
        "land_fraction": _safe_float(detection.get("land_fraction"), 1.0),
        "pressure_anomaly_pa": _safe_float(
            detection.get("pressure_anomaly_pa")
        ),
        "midlevel_rh_percent": _safe_float(
            detection.get("midlevel_rh_percent")
        ),
        "shear_200_850_ms": _safe_float(
            detection.get("shear_200_850_ms"), 99.0
        ),
        "vorticity_850_s1": raw_vorticity,
        "cyclonic_vorticity_850_s1": (
            raw_vorticity if latitude >= 0.0 else -raw_vorticity
        ),
        "surface_temperature_k": _safe_float(
            detection.get("surface_temperature_k")
        ),
        "matched_distance_km": _safe_float(nearest_distance_km, 2000.0),
        "matched_age_hours": _safe_float(nearest_age_hours, 999.0),
    }

    for name in V3D_NAMES + V2D_NAMES:
        value = nested_xai.get(name)
        if value is None:
            value = detection.get(f"xai_mean_{name}")
        if value is None:
            value = detection.get(f"xai_{name}")
        features[f"xai_{name}"] = _safe_float(value)
        features[f"xai_mean_{name}"] = _safe_float(value)

    if model is not None:
        medians = model.get("imputation_medians", {}) or {}
        for feature_name in model.get("feature_names", []):
            if feature_name not in features:
                alias = feature_name.replace("xai_mean_", "xai_")
                if alias in features:
                    features[feature_name] = features[alias]
                else:
                    features[feature_name] = _safe_float(
                        medians.get(feature_name), 0.0
                    )
    return features

def predict_post_classifier_probability(
    model: Mapping[str, Any],
    features: Mapping[str, float],
) -> float:
    names = list(model["feature_names"])
    coefficients = model["coefficients"]
    means = model.get("means", {}) or {}
    scales = model.get("scales", {}) or {}
    medians = model.get("imputation_medians", {}) or {}
    logit = float(model["intercept"])

    for index, name in enumerate(names):
        raw_value = features.get(name, medians.get(name, means.get(name, 0.0)))
        value = _safe_float(raw_value, _safe_float(medians.get(name)))
        mean = _safe_float(means.get(name))
        scale = max(abs(_safe_float(scales.get(name), 1.0)), 1e-12)
        standardized = (value - mean) / scale
        coefficient = (
            _safe_float(coefficients.get(name))
            if isinstance(coefficients, dict)
            else _safe_float(coefficients[index])
        )
        logit += coefficient * standardized

    if not math.isfinite(logit):
        raise ValueError("Learned post-classifier produced a non-finite logit.")
    logit = max(-40.0, min(40.0, logit))
    probability = 1.0 / (1.0 + math.exp(-logit))
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Learned post-classifier produced an invalid probability.")
    return probability



def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _soft_pass(value: float, threshold: float, scale: float, *, higher_is_better: bool) -> float:
    """Smooth 0-1 support around a physical threshold."""
    scale = max(abs(scale), 1e-12)
    signed = (value - threshold) / scale
    if not higher_is_better:
        signed = -signed
    signed = max(-20.0, min(20.0, signed))
    return 1.0 / (1.0 + math.exp(-signed))


def compute_xai_environment_support(detection: Mapping[str, Any]) -> Dict[str, float]:
    """Build variable confidence from neural agreement, physics margins and XAI.

    This is a fallback score, not a calibrated probability. Unlike the old fixed
    0.60/0.88 constants, it varies by hotspot and rewards coherent evidence.
    """
    lat = _safe_float(detection.get("latitude"))
    vort_raw = _safe_float(detection.get("vorticity_850_s1"))
    cyclonic_vort = vort_raw if lat >= 0.0 else -vort_raw

    land = _safe_float(detection.get("land_fraction"), 1.0)
    rh = _safe_float(detection.get("midlevel_rh_percent"))
    shear = _safe_float(detection.get("shear_200_850_ms"), 99.0)
    pressure = _safe_float(detection.get("pressure_anomaly_pa"), 9999.0)
    sst = _safe_float(detection.get("surface_temperature_k"))

    supports = {
        "oceanic": _soft_pass(land, 0.15, 0.08, higher_is_better=False),
        "moist": _soft_pass(rh, 55.0, 10.0, higher_is_better=True),
        "low_shear": _soft_pass(shear, 20.0, 5.0, higher_is_better=False),
        "cyclonic_vorticity": _soft_pass(cyclonic_vort, 7.5e-6, 1.5e-5, higher_is_better=True),
        "low_pressure": _soft_pass(pressure, 0.0, 250.0, higher_is_better=False),
        "warm_surface": _soft_pass(sst, 299.0, 1.5, higher_is_better=True),
    }
    environment_support = sum(supports.values()) / len(supports)

    xai = detection.get("xai_mean_attribution", {}) or {}
    xai_values = {name: max(0.0, _safe_float(xai.get(name))) for name in V3D_NAMES + V2D_NAMES}
    total_xai = sum(xai_values.values())
    genesis_xai_names = {"pressfc", "rhprs", "absvprs", "ugrdprs", "vgrdprs", "vvelprs", "tmpsfc", "capesfc"}
    xai_support = (
        sum(value for name, value in xai_values.items() if name in genesis_xai_names) / total_xai
        if total_xai > 0.0 else 0.5
    )

    ensemble = _clip01(_safe_float(detection.get("confidence")))
    cnn = _clip01(_safe_float(detection.get("cnn_confidence")))
    vit = _clip01(_safe_float(detection.get("vit_confidence")))
    expert_agreement = _clip01(1.0 - abs(cnn - vit))
    neural_support = 0.50 * ensemble + 0.25 * cnn + 0.25 * vit

    # Physics is primary, neural/XAI evidence refines it. This is deliberately
    # conservative and is not presented as a calibrated formation probability.
    hybrid_support = _clip01(
        0.50 * environment_support
        + 0.30 * neural_support
        + 0.12 * xai_support
        + 0.08 * expert_agreement
    )
    return {
        "environment_support": environment_support,
        "xai_support": xai_support,
        "neural_support": neural_support,
        "expert_agreement": expert_agreement,
        "hybrid_support": hybrid_support,
        **{f"support_{key}": value for key, value in supports.items()},
    }


def learned_model_is_usable(model: Optional[Mapping[str, Any]]) -> tuple[bool, str]:
    """Reject an unvalidated learned post-classifier and force rule fallback."""
    if model is None:
        return False, "learned model file was not loaded"
    metrics = model.get("validation_metrics", {}) or {}
    n = int(metrics.get("n_samples", model.get("training_samples", 0)) or 0)
    precision = _safe_float(metrics.get("precision"), -1.0)
    csi = _safe_float(metrics.get("csi"), -1.0)
    baseline_precision = _safe_float(metrics.get("rules_precision"), 0.0)
    baseline_csi = _safe_float(metrics.get("rules_csi"), 0.0)
    if n < int(model.get("minimum_validation_samples", 100)):
        return False, f"validation sample count {n} is below the required minimum"
    if precision < baseline_precision or csi < baseline_csi:
        return False, "learned validation precision/CSI did not beat the six-condition fallback"
    if precision < float(model.get("minimum_validation_precision", 0.25)):
        return False, "learned validation precision is below the safety floor"
    return True, "validated learned model"


def learned_prediction_is_in_distribution(
    model: Mapping[str, Any], features: Mapping[str, float]
) -> tuple[bool, str]:
    """Reject only clearly nonsensical learned inputs.

    A single meteorological feature outside five standard deviations is common
    in extreme weather and should not disable the model.  We fall back only
    when several features are extreme or an input is non-finite.
    """
    means = model.get("means", {}) or {}
    scales = model.get("scales", {}) or {}
    max_z = float(model.get("maximum_absolute_z", 5.0))
    extreme: List[str] = []
    for name in model.get("feature_names", []):
        value = _safe_float(features.get(name), float("nan"))
        if not math.isfinite(value):
            return False, f"non-finite learned feature: {name}"
        scale = abs(_safe_float(scales.get(name), 0.0))
        if scale <= 1e-12:
            continue
        z = abs((value - _safe_float(means.get(name))) / scale)
        if z > max_z:
            extreme.append(f"{name}={z:.1f}σ")

    allowed_extreme = max(2, int(math.ceil(0.15 * len(model.get("feature_names", [])))))
    if len(extreme) > allowed_extreme:
        return False, (
            f"{len(extreme)} learned features are far outside training range: "
            + ", ".join(extreme[:4])
        )
    if extreme:
        return True, "learned model used with limited extrapolation: " + ", ".join(extreme[:4])
    return True, "features are inside the learned training range"

def deduplicate_matched_systems(
    detections: Sequence[Dict[str, Any]],
    same_system_radius_km: float,
) -> List[Dict[str, Any]]:
    """Keep one best hotspot per matched storm and suppress nearby duplicates."""
    ranked = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
    retained: List[Dict[str, Any]] = []
    used_systems = set()
    for candidate in ranked:
        system_key = candidate.get("matched_system_id") or candidate.get("matched_system_name")
        if system_key and system_key not in {"UNKNOWN", "UNNAMED", "TROPICAL"}:
            if system_key in used_systems:
                continue
            used_systems.add(system_key)
        duplicate = False
        for existing in retained:
            if haversine_distance_km(
                float(candidate["latitude"]), float(candidate["longitude"]),
                float(existing["latitude"]), float(existing["longitude"]),
            ) <= same_system_radius_km:
                ckey = candidate.get("matched_system_id") or candidate.get("matched_system_name")
                ekey = existing.get("matched_system_id") or existing.get("matched_system_name")
                if ckey and ekey and ckey == ekey:
                    duplicate = True
                    break
        if not duplicate:
            retained.append(candidate)
    return retained


def classify_disturbance(
    detection: Dict[str, Any],
    active_systems: Sequence[Dict[str, Any]],
    initialization_time: datetime,
    *,
    classification_mode: str = "rules",
    post_classifier_model: Optional[Mapping[str, Any]] = None,
    stale_track_radius_km: float = 500.0,
) -> Dict[str, Any]:
    """
    Assign a conservative operational class to a retained hotspot.

    Important:
    - The model may issue an unverified genesis candidate before an official invest exists.
    - Track data are used to identify already-active systems, not to veto model discovery.
    - Learned XAI/environment classification is used only after validation quality and
      out-of-distribution checks; otherwise the six-condition rules are restored.
    - Classification confidence is a variable evidence score, not a calibrated probability.
    """
    latitude = float(detection["latitude"])
    longitude = float(detection["longitude"])

    land_fraction = float(detection.get("land_fraction", 1.0))
    pressure_anomaly = float(
        detection.get("pressure_anomaly_pa", 9999.0)
    )
    midlevel_rh = float(detection.get("midlevel_rh_percent", 0.0))
    shear = float(detection.get("shear_200_850_ms", 999.0))
    vort_raw = float(detection.get("vorticity_850_s1", 0.0))
    # Cyclonic sign reverses across the equator: positive in NH, negative in SH.
    vort = vort_raw if latitude >= 0.0 else -vort_raw
    sst = float(detection.get("surface_temperature_k", 0.0))

    favorable_checks = {
        "oceanic": land_fraction <= 0.15,
        "moist": midlevel_rh >= 55.0,
        "low_shear": shear <= 20.0,
        "cyclonic_vorticity": vort >= 7.5e-6,
        "low_pressure": pressure_anomaly <= 0.0,
        "warm_surface": sst >= 299.0,
    }
    favorable_count = sum(favorable_checks.values())
    support = compute_xai_environment_support(detection)
    detection.update({
        "favorable_environmental_checks": favorable_checks,
        "favorable_environmental_count": favorable_count,
        "cyclonic_vorticity_850_s1": vort,
        "environment_support_percent": round(100.0 * support["environment_support"], 2),
        "xai_support_percent": round(100.0 * support["xai_support"], 2),
        "neural_support_percent": round(100.0 * support["neural_support"], 2),
        "hybrid_support_percent": round(100.0 * support["hybrid_support"], 2),
    })
    rule_confidence = support["hybrid_support"]

    nearest_system: Optional[Dict[str, Any]] = None
    nearest_distance_km = float("inf")
    nearest_age_hours = float("inf")

    for system in active_systems:
        distance_km = haversine_distance_km(
            latitude,
            longitude,
            float(system["latitude"]),
            float(system["longitude"]),
        )
        age_hours = abs(
            (initialization_time - system["time"]).total_seconds()
        ) / 3600.0

        comparison_score = distance_km + 5.0 * age_hours
        current_score = nearest_distance_km + 5.0 * nearest_age_hours
        if comparison_score < current_score:
            nearest_system = system
            nearest_distance_km = distance_km
            nearest_age_hours = age_hours

    def attach_match() -> Dict[str, Any]:
        if nearest_system is None:
            return {}
        return {
            "matched_system_name": nearest_system["name"],
            "matched_system_id": nearest_system.get("storm_id"),
            "matched_system_status": nearest_system["status"],
            "matched_system_source": nearest_system.get("source"),
            "matched_system_basin": nearest_system.get("basin"),
            "matched_system_distance_km": round(nearest_distance_km, 1),
            "matched_system_age_hours": round(nearest_age_hours, 1),
        }

    learned_probability: Optional[float] = None
    learned_features: Optional[Dict[str, float]] = None
    learned_ok, learned_gate_reason = learned_model_is_usable(post_classifier_model)
    if classification_mode in {"learned", "auto"} and learned_ok and post_classifier_model is not None:
        learned_features = build_post_classifier_features(
            detection, nearest_distance_km, nearest_age_hours, post_classifier_model
        )
        in_distribution, distribution_reason = learned_prediction_is_in_distribution(
            post_classifier_model, learned_features
        )
        if in_distribution:
            learned_probability = predict_post_classifier_probability(
                post_classifier_model, learned_features
            )
            detection["learned_genesis_probability"] = learned_probability
            detection["learned_genesis_probability_percent"] = round(learned_probability * 100.0, 2)
            detection["post_classifier_used"] = True
            detection["classification_source"] = "LEARNED_XAI_POST_CLASSIFIER"
            detection["post_classification_method"] = "validated_learned_xai_environment"
            detection["learned_model_gate"] = f"{learned_gate_reason}; {distribution_reason}"
        else:
            detection["post_classifier_used"] = False
            detection["classification_source"] = "SIX_CONDITION_XAI_FALLBACK"
            detection["post_classification_method"] = "six_condition_xai_fallback"
            detection["learned_model_gate"] = distribution_reason
    else:
        detection["post_classifier_used"] = False
        detection["classification_source"] = "SIX_CONDITION_XAI_FALLBACK"
        detection["post_classification_method"] = "six_condition_xai_fallback"
        detection["learned_model_gate"] = learned_gate_reason

    if nearest_system is not None:
        status = str(nearest_system["status"]).upper()
        name = nearest_system["name"]

        # Current official cyclone position.
        if (
            nearest_distance_km <= 600.0
            and nearest_age_hours <= 36.0
            and status in ACTIVE_CYCLONE_STATUSES
        ):
            detection.update(
                {
                    "system_class": "ACTIVE_CYCLONE",
                    "classification_confidence": 0.95,
                    "classification_reason": (
                        f"Within {nearest_distance_km:.0f} km of active "
                        f"tracked system {name} ({status}); track age "
                        f"{nearest_age_hours:.1f} h."
                    ),
                    **attach_match(),
                }
            )
            return detection

        # A recent depression bulletin should not be reported as fresh genesis.
        # Some OSPO feeds encode depression/deep-depression systems as DB.
        if (
            nearest_distance_km <= 400.0
            and nearest_age_hours <= 48.0
            and status == "DB"
            and midlevel_rh >= 60.0
            and shear <= 25.0
            and vort >= 7.5e-6
        ):
            detection.update(
                {
                    "system_class": "ACTIVE_DEPRESSION",
                    "classification_confidence": 0.90,
                    "classification_reason": (
                        f"Matched recent depression/disturbance bulletin {name} "
                        f"({status}) within {nearest_distance_km:.0f} km; moist, "
                        "cyclonic and low-shear structure indicates an existing "
                        "organized depression rather than new genesis."
                    ),
                    **attach_match(),
                }
            )
            return detection

        # Invest, tropical disturbance or potential cyclone.
        if (
            nearest_distance_km <= 650.0
            and nearest_age_hours <= 48.0
            and status in INVEST_OR_DISTURBANCE_STATUSES
        ):
            if learned_probability is not None:
                learned_threshold = float(post_classifier_model.get("positive_threshold", 0.50))
                if learned_probability >= learned_threshold:
                    system_class = "NEW_GENESIS_CANDIDATE"
                    confidence = learned_probability
                    reason = (
                        f"Learned post-classifier probability {learned_probability:.1%} "
                        f"exceeded {learned_threshold:.1%} for matched invest/disturbance "
                        f"{name} ({status}) within {nearest_distance_km:.0f} km. "
                        "The classifier uses historical environmental, neural and XAI features."
                    )
                else:
                    system_class = "INVEST_OR_DEVELOPING_SYSTEM"
                    confidence = 1.0 - learned_probability
                    reason = (
                        f"Learned post-classifier probability {learned_probability:.1%} "
                        f"was below {learned_threshold:.1%} for matched invest/disturbance "
                        f"{name} ({status})."
                    )
            elif classification_mode == "learned":
                system_class = "INVEST_OR_DEVELOPING_SYSTEM"
                confidence = 0.50
                reason = (
                    "Learned classification was requested, but no valid post-classifier "
                    "model was loaded; no genesis claim was made."
                )
            elif favorable_count >= 4:
                system_class = "NEW_GENESIS_CANDIDATE"
                confidence = max(0.55, min(0.95, rule_confidence))
                reason = (
                    f"Matched recent invest/disturbance {name} ({status}) "
                    f"within {nearest_distance_km:.0f} km and "
                    f"{favorable_count}/6 environmental checks passed."
                )
            else:
                system_class = "INVEST_OR_DEVELOPING_SYSTEM"
                confidence = max(0.50, min(0.85, rule_confidence))
                reason = (
                    f"Matched recent invest/disturbance {name} ({status}) "
                    f"within {nearest_distance_km:.0f} km, but only "
                    f"{favorable_count}/6 genesis-environment checks passed."
                )

            detection.update(
                {
                    "system_class": system_class,
                    "classification_confidence": confidence,
                    "classification_reason": reason,
                    **attach_match(),
                }
            )
            return detection

        # Explicitly decaying/remnant status only.
        if (
            nearest_distance_km <= stale_track_radius_km
            and nearest_age_hours <= 72.0
            and status in DECAYING_SYSTEM_STATUSES
        ):
            detection.update(
                {
                    "system_class": "RECENT_DECAYING_SYSTEM",
                    "classification_confidence": 0.88,
                    "classification_reason": (
                        f"Within {nearest_distance_km:.0f} km of recent "
                        f"decaying/remnant system {name} ({status}); track "
                        f"age {nearest_age_hours:.1f} h."
                    ),
                    **attach_match(),
                }
            )
            return detection

        # A stale cyclone fix should not automatically be called decaying.
        if (
            nearest_distance_km <= stale_track_radius_km
            and nearest_age_hours <= 72.0
            and status in ACTIVE_CYCLONE_STATUSES
        ):
            detection.update(
                {
                    "system_class": "RECENT_TRACKED_CYCLONE",
                    "classification_confidence": 0.82,
                    "classification_reason": (
                        f"Near the latest available track of {name} "
                        f"({status}), but the position is "
                        f"{nearest_age_hours:.1f} h old; current lifecycle "
                        "cannot be determined from this stale fix."
                    ),
                    **attach_match(),
                }
            )
            return detection

    # Coastal/land-dominated monsoon circulation.
    if (
        land_fraction >= 0.20
        and midlevel_rh >= 60.0
    ):
        detection.update(
            {
                "system_class": "MONSOON_OR_LAND_LOW",
                "classification_confidence": 0.72,
                "classification_reason": (
                    "Strong moist circulation with substantial land/coastal "
                    "influence and no qualifying active/invest match."
                ),
            }
        )
        return detection

    # The research objective is early discovery. Lack of an official invest
    # changes verification_status, but does not veto a well-supported model call.
    detection["verification_status"] = "UNVERIFIED"

    if learned_probability is not None and post_classifier_model is not None:
        positive_threshold = float(post_classifier_model.get("positive_threshold", 0.50))
        high_threshold = float(post_classifier_model.get("high_priority_threshold", max(0.70, positive_threshold + 0.15)))
        if learned_probability >= high_threshold:
            detection.update({
                "system_class": "HIGH_PRIORITY_MODEL_GENESIS_CANDIDATE",
                "classification_confidence": learned_probability,
                "classification_reason": (
                    f"Validated learned XAI/environment classifier produced "
                    f"{learned_probability:.1%}, above the high-priority threshold "
                    f"{high_threshold:.1%}; no official track match is currently available."
                ),
            })
            return detection
        if learned_probability >= positive_threshold:
            detection.update({
                "system_class": "MODEL_GENESIS_CANDIDATE",
                "classification_confidence": learned_probability,
                "classification_reason": (
                    f"Validated learned XAI/environment classifier produced "
                    f"{learned_probability:.1%}, above the genesis threshold "
                    f"{positive_threshold:.1%}; candidate remains operationally unverified."
                ),
            })
            return detection
        # A validated negative learned result may override rules only when it is
        # decisive; borderline output falls back to the six-condition method.
        negative_threshold = float(post_classifier_model.get("negative_threshold", max(0.20, positive_threshold - 0.20)))
        if learned_probability <= negative_threshold:
            detection.update({
                "system_class": "LOW_SUPPORT_DISTURBANCE",
                "classification_confidence": 1.0 - learned_probability,
                "classification_reason": (
                    f"Validated learned XAI/environment classifier produced only "
                    f"{learned_probability:.1%}, below the negative threshold "
                    f"{negative_threshold:.1%}."
                ),
            })
            return detection
        detection["post_classifier_used"] = False
        detection["classification_source"] = "SIX_CONDITION_XAI_FALLBACK_BORDERLINE"
        detection["post_classification_method"] = "six_condition_xai_fallback_borderline_learned_result"

    if favorable_count == 6:
        detection.update({
            "system_class": "HIGH_PRIORITY_MODEL_GENESIS_CANDIDATE",
            "classification_confidence": max(0.70, min(0.95, rule_confidence)),
            "classification_reason": (
                "All 6/6 genesis-environment checks passed with strong neural/XAI "
                "support. No official track match exists, so the candidate is model-"
                "identified and operationally unverified."
            ),
        })
        return detection

    if favorable_count == 5:
        detection.update({
            "system_class": "MODEL_GENESIS_CANDIDATE",
            "classification_confidence": max(0.62, min(0.90, rule_confidence)),
            "classification_reason": (
                "5/6 genesis-environment checks passed with neural/XAI support. "
                "No official track match exists; this is an unverified early model candidate."
            ),
        })
        return detection

    if favorable_count == 4:
        detection.update({
            "system_class": "POSSIBLE_MODEL_GENESIS_CANDIDATE",
            "classification_confidence": max(0.52, min(0.80, rule_confidence)),
            "classification_reason": (
                "4/6 genesis-environment checks passed. The hotspot is retained as a "
                "possible unverified genesis candidate rather than being suppressed."
            ),
        })
        return detection

    if land_fraction <= 0.15 and midlevel_rh >= 50.0 and vort >= 4.0e-6:
        detection.update({
            "system_class": "UNVERIFIED_TROPICAL_DISTURBANCE",
            "classification_confidence": max(0.40, min(0.70, rule_confidence)),
            "classification_reason": (
                "Oceanic moist cyclonic neural signal without sufficient multi-condition "
                "support for a genesis-candidate class."
            ),
        })
        return detection

    detection.update(
        {
            "system_class": "UNCLASSIFIED_DISTURBANCE",
            "classification_confidence": 0.40,
            "classification_reason": (
                "Strong neural signal, but neither official track matching "
                "nor the conservative environmental rules support a more "
                "specific class."
            ),
        }
    )
    return detection


# =============================================================================
# INFERENCE
# =============================================================================

def build_candidate_locations(
    ds: xr.Dataset,
    config: RuntimeConfig,
) -> List[Tuple[int, int, float, float]]:
    lats = np.asarray(ds["lat"].values, dtype=np.float32)
    lons = np.asarray(ds["lon"].values, dtype=np.float32)
    landmask = np.asarray(ds["landmask"].values, dtype=np.float32)
    surface_temperature = np.asarray(ds["tmpsfc"].values, dtype=np.float32)

    candidates: List[Tuple[int, int, float, float]] = []

    for lat_index, latitude in enumerate(lats):
        if latitude < config.min_latitude or latitude > config.max_latitude:
            continue

        if lat_index - HALF_PATCH < 0 or lat_index + HALF_PATCH >= len(lats):
            continue

        for lon_index, longitude_360 in enumerate(lons):
            if float(landmask[lat_index, lon_index]) > config.maximum_land_fraction:
                continue

            if (
                config.minimum_surface_temperature_k is not None
                and float(surface_temperature[lat_index, lon_index])
                < config.minimum_surface_temperature_k
            ):
                continue

            longitude_export = (
                float(longitude_360)
                if longitude_360 <= 180.0
                else float(longitude_360 - 360.0)
            )
            candidates.append(
                (
                    lat_index,
                    lon_index,
                    float(latitude),
                    longitude_export,
                )
            )

    return candidates


def predict_optional_expert(
    model: Optional[tf.keras.Model],
    x_batch: np.ndarray,
    batch_size: int,
) -> Optional[np.ndarray]:
    if model is None:
        return None
    result = model.predict(
        x_batch,
        batch_size=batch_size,
        verbose=0,
    )
    return np.asarray(result, dtype=np.float32).reshape(-1)


def run_sliding_window_inference(
    ds: xr.Dataset,
    ensemble_model: tf.keras.Model,
    cnn_expert: Optional[tf.keras.Model],
    vit_expert: Optional[tf.keras.Model],
    *,
    threshold: float,
    means: np.ndarray,
    stds: np.ndarray,
    config: RuntimeConfig,
    active_systems: Sequence[Dict[str, Any]],
    initialization_time: datetime,
) -> List[Dict[str, Any]]:
    """
    Performs global 21 × 21 sliding-window inference.

    The same atmospheric analysis field is used for all horizon-specific models.
    Lead time is determined by the model checkpoint, not by shifting the input
    field to f024/f048.
    """
    ds = interpolate_to_training_grid(ds)
    full_grid = construct_normalized_grid(ds, means, stds)

    # Longitude wrap only. Latitude is not cyclic.
    padded_grid = np.pad(
        full_grid,
        ((0, 0), (HALF_PATCH, HALF_PATCH), (0, 0)),
        mode="wrap",
    )

    candidates = build_candidate_locations(ds, config)
    logging.info("Evaluating %d oceanic candidate patches.", len(candidates))

    raw_detections: List[Dict[str, Any]] = []
    num_batches = math.ceil(len(candidates) / config.batch_size)

    for start in tqdm(
        range(0, len(candidates), config.batch_size),
        total=num_batches,
        desc="Evaluating candidate batches",
    ):
        batch_locations = candidates[start : start + config.batch_size]
        patches: List[np.ndarray] = []
        metadata: List[Tuple[int, int, float, float]] = []

        for lat_index, lon_index, latitude, longitude_export in batch_locations:
            padded_lon_index = lon_index + HALF_PATCH

            patch = padded_grid[
                lat_index - HALF_PATCH : lat_index + HALF_PATCH + 1,
                padded_lon_index - HALF_PATCH : padded_lon_index + HALF_PATCH + 1,
                :,
            ]

            if patch.shape != (PATCH_SIZE, PATCH_SIZE, EXPECTED_CHANNELS):
                raise RuntimeError(
                    f"Unexpected patch shape {patch.shape} at "
                    f"lat={latitude}, lon={longitude_export}"
                )

            patches.append(patch)
            metadata.append(
                (lat_index, lon_index, latitude, longitude_export)
            )

        if not patches:
            continue

        x_batch = np.asarray(patches, dtype=np.float32)

        ensemble_probabilities = np.asarray(
            ensemble_model.predict(
                x_batch,
                batch_size=config.batch_size,
                verbose=0,
            ),
            dtype=np.float32,
        ).reshape(-1)

        cnn_probabilities = predict_optional_expert(
            cnn_expert,
            x_batch,
            config.batch_size,
        )
        vit_probabilities = predict_optional_expert(
            vit_expert,
            x_batch,
            config.batch_size,
        )

        for item_index, ensemble_probability in enumerate(ensemble_probabilities):
            if float(ensemble_probability) < threshold:
                continue

            (
                lat_index,
                lon_index,
                latitude,
                longitude,
            ) = metadata[item_index]
            input_tensor = np.expand_dims(x_batch[item_index], axis=0)

            (
                top_variable,
                mean_attribution,
                total_column_attribution,
            ) = compute_gradient_attribution(
                ensemble_model,
                input_tensor,
            )

            environmental_metrics = calculate_environmental_metrics(
                ds,
                lat_index=lat_index,
                lon_index=lon_index,
            )

            detection: Dict[str, Any] = {
                "latitude": latitude,
                "longitude": longitude,
                **environmental_metrics,
                "confidence": float(ensemble_probability),
                "confidence_percent": round(
                    float(ensemble_probability) * 100.0,
                    2,
                ),
                "top_driver": top_variable,
                "xai_method": "mean_absolute_input_gradient",
                "xai_mean_attribution": mean_attribution,
                "xai_total_column_attribution": total_column_attribution,
            }

            if cnn_probabilities is not None:
                detection["cnn_confidence"] = float(
                    cnn_probabilities[item_index]
                )

            if vit_probabilities is not None:
                detection["vit_confidence"] = float(
                    vit_probabilities[item_index]
                )

            available_experts = [
                value for value in (
                    detection.get("cnn_confidence"),
                    detection.get("vit_confidence"),
                )
                if value is not None
            ]
            if available_experts and min(available_experts) < config.min_expert_confidence:
                continue

            detection["real_world_confidence"] = REAL_WORLD_PRECISION_REFERENCE
            detection["real_world_confidence_percent"] = round(
                REAL_WORLD_PRECISION_REFERENCE * 100.0, 2
            )
            detection["real_world_confidence_basis"] = (
                "Empirical precision from strict 2026 FNL verification "
                "(16 hits / 52 retained disturbances). Retrospective reference "
                "rate, not a calibrated per-storm probability."
            )
            detection["empirical_pod_reference"] = REAL_WORLD_POD_REFERENCE
            detection["empirical_csi_reference"] = REAL_WORLD_CSI_REFERENCE

            raw_detections.append(detection)

    logging.info(
        "Generated %d detections before spatial NMS.",
        len(raw_detections),
    )

    retained = apply_spatial_nms(
        raw_detections,
        distance_threshold_km=config.nms_radius_km,
    )
    retained = sorted(
        retained,
        key=lambda item: float(item["confidence"]),
        reverse=True,
    )
    post_classifier_model = load_post_classifier_model(config.post_classifier_model)
    classified = [
        classify_disturbance(
            detection=item,
            active_systems=active_systems,
            initialization_time=initialization_time,
            classification_mode=config.classification_mode,
            post_classifier_model=post_classifier_model,
            stale_track_radius_km=config.stale_track_radius_km,
        )
        for item in retained
    ]
    classified = deduplicate_matched_systems(
        classified, same_system_radius_km=config.same_system_radius_km
    )
    classified = sorted(
        classified, key=lambda item: float(item.get("confidence", 0.0)), reverse=True
    )
    return classified[: config.max_detections]


# =============================================================================
# EXPORT
# =============================================================================

def flatten_xai_properties(
    prefix: str,
    values: Mapping[str, float],
) -> Dict[str, float]:
    return {
        f"{prefix}_{key}": float(value)
        for key, value in values.items()
    }


def export_geojson(
    predictions: Sequence[Dict[str, Any]],
    *,
    horizon: int,
    initialization_time: datetime,
    threshold: float,
    source_product: str,
) -> Path:
    timestamp = initialization_time.strftime("%Y%m%d_%H%M")
    output_path = OUTPUT_DIR / f"genesis_forecast_{horizon}h_{timestamp}.geojson"
    latest_path = OUTPUT_DIR / f"latest_{horizon}h.geojson"

    features: List[Dict[str, Any]] = []

    for prediction in predictions:
        properties: Dict[str, Any] = {
            "confidence": prediction["confidence"],
            "confidence_percent": prediction["confidence_percent"],
            "raw_model_score": prediction["confidence"],
            "raw_model_score_percent": prediction["confidence_percent"],
            "real_world_confidence": prediction["real_world_confidence"],
            "real_world_confidence_percent": prediction["real_world_confidence_percent"],
            "real_world_confidence_basis": prediction["real_world_confidence_basis"],
            "empirical_pod_reference": prediction["empirical_pod_reference"],
            "empirical_csi_reference": prediction["empirical_csi_reference"],
            "lead_time_hrs": horizon,
            "validation_threshold": threshold,
            "top_driver": prediction["top_driver"],
            "xai_method": prediction["xai_method"],
            "source_product": source_product,
            "initialization_time_utc": initialization_time.isoformat(),
            "system_class": prediction.get(
                "system_class", "UNCLASSIFIED_DISTURBANCE"
            ),
            "classification_confidence": prediction.get(
                "classification_confidence", 0.0
            ),
            "classification_confidence_percent": round(
                float(prediction.get("classification_confidence", 0.0))
                * 100.0,
                2,
            ),
            "classification_reason": prediction.get(
                "classification_reason", ""
            ),
            "land_fraction": prediction.get("land_fraction"),
            "pressure_anomaly_pa": prediction.get("pressure_anomaly_pa"),
            "midlevel_rh_percent": prediction.get("midlevel_rh_percent"),
            "shear_200_850_ms": prediction.get("shear_200_850_ms"),
            "vorticity_850_s1": prediction.get("vorticity_850_s1"),
            "surface_temperature_k": prediction.get(
                "surface_temperature_k"
            ),
        }

        for optional_key in (
            "matched_system_name",
            "matched_system_status",
            "matched_system_distance_km",
            "matched_system_age_hours",
        ):
            if optional_key in prediction:
                properties[optional_key] = prediction[optional_key]

        if "cnn_confidence" in prediction:
            properties["cnn_confidence"] = prediction["cnn_confidence"]

        if "vit_confidence" in prediction:
            properties["vit_confidence"] = prediction["vit_confidence"]

        properties.update(
            flatten_xai_properties(
                "xai_mean",
                prediction["xai_mean_attribution"],
            )
        )
        properties.update(
            flatten_xai_properties(
                "xai_column_total",
                prediction["xai_total_column_attribution"],
            )
        )

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
        "metadata": {
            "initialization_time_utc": initialization_time.isoformat(),
            "lead_time_hrs": horizon,
            "threshold": threshold,
            "source_product": source_product,
            "training_domain_note": (
                "Model input uses native NCEP FNL fields. Post-class labels "
                "are rule-based diagnostics, not calibrated probabilities."
            ),
            "channel_schema": CHANNEL_SCHEMA,
        },
        "features": features,
    }

    for path in (output_path, latest_path):
        with path.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)

    logging.info("GeoJSON exported to %s", output_path)
    logging.info("Latest GeoJSON updated at %s", latest_path)
    return output_path


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

def floor_to_previous_synoptic_cycle(now: datetime) -> datetime:
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    cycle_hour = (now.hour // 6) * 6
    return now.replace(
        hour=cycle_hour,
        minute=0,
        second=0,
        microsecond=0,
    )


def acquire_latest_analysis(
    session: requests.Session,
) -> Tuple[datetime, Path]:
    """
    Resolve the newest available native NCEP FNL analysis.

    FNL is delayed rather than truly real-time, so the archive is probed
    backward over recent six-hourly cycles until an available file is found.
    """
    base_time = floor_to_previous_synoptic_cycle(datetime.now(timezone.utc))

    for lookback_steps in range(0, 30 * 4):
        target_time = base_time - timedelta(hours=lookback_steps * 6)
        file_path = download_fnl_analysis(target_time, session=session)
        if file_path is not None:
            return target_time, file_path

    raise RuntimeError(
        "No recent NCEP FNL analysis could be acquired from the archive."
    )


def execute_pipeline_cycle(config: RuntimeConfig) -> None:
    validate_channel_schema_file(config.channel_names_file)
    validate_training_metadata(
        config.training_metadata_file,
        allow_domain_shift=True,
    )

    means, stds = load_normalization_statistics(
        config.means_file,
        config.stds_file,
    )
    thresholds = load_operational_thresholds(config.threshold_file)

    missing_thresholds = [
        leadtime for leadtime in config.lead_times
        if leadtime not in thresholds
    ]
    if missing_thresholds:
        raise ValueError(
            f"Missing validation threshold(s) for lead times: "
            f"{missing_thresholds}"
        )

    session = get_requests_session(insecure_ssl=config.insecure_ssl)
    initialization_time, analysis_path = acquire_latest_analysis(session)
    ds = preprocess_raw_grib(analysis_path)
    active_systems = load_active_systems(config.active_systems_json)

    for leadtime in config.lead_times:
        logging.info(
            "================ %d-HOUR GENESIS MODEL ================",
            leadtime,
        )

        ensemble_model, cnn_expert, vit_expert = load_leadtime_models(
            leadtime
        )
        threshold = float(config.alert_threshold)

        detections = run_sliding_window_inference(
            ds,
            ensemble_model,
            cnn_expert,
            vit_expert,
            threshold=threshold,
            means=means,
            stds=stds,
            config=config,
            active_systems=active_systems,
            initialization_time=initialization_time,
        )

        for detection in detections:
            detection["lead_time_hrs"] = leadtime

        export_geojson(
            detections,
            horizon=leadtime,
            initialization_time=initialization_time,
            threshold=threshold,
            source_product="NCEP FNL native 1-degree analysis",
        )


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def parse_lead_times(raw: str) -> Tuple[int, ...]:
    values = tuple(sorted({int(item.strip()) for item in raw.split(",") if item.strip()}))
    invalid = [value for value in values if value not in {24, 48, 72}]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Supported lead times are 24, 48 and 72; invalid: {invalid}"
        )
    if not values:
        raise argparse.ArgumentTypeError("At least one lead time is required.")
    return values


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Operational tropical cyclogenesis inference using exact "
            "137-channel mapping and horizon-specific models."
        )
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Execute one forecast cycle and exit.",
    )
    parser.add_argument(
        "--lead-times",
        type=parse_lead_times,
        default=(24, 48, 72),
        help="Comma-separated horizons. Default: 24,48,72",
    )
    parser.add_argument(
        "--allow-domain-shift",
        action="store_true",
        help=(
            "Allow NCAR-trained models to run on operational GDAS/GFS fields. "
            "Use only after validation or with the limitation documented."
        ),
    )
    parser.add_argument(
        "--insecure-ssl",
        action="store_true",
        help=(
            "Disable SSL certificate verification for an institutional proxy. "
            "Not recommended."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Inference batch size. Default: 256",
    )
    parser.add_argument(
        "--nms-radius-km",
        type=float,
        default=550.0,
        help="Great-circle NMS radius in kilometres. Default: 550",
    )
    parser.add_argument(
        "--min-expert-confidence",
        type=float,
        default=0.45,
        help="Minimum CNN and ViT expert score. Default: 0.45",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=2,
        help="Maximum retained global disturbances per horizon. Default: 2",
    )
    parser.add_argument(
        "--active-systems-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON containing active/recent storm positions for "
            "post-classification."
        ),
    )
    parser.add_argument(
        "--classification-mode",
        choices=("rules", "learned", "auto"),
        default="auto",
        help="Post-classification mode. auto uses learned JSON when present, otherwise rules.",
    )
    parser.add_argument(
        "--post-classifier-model",
        type=Path,
        default=MODELS_DIR / "genesis_post_classifier.json",
        help="Safe JSON logistic post-classifier trained on historical hit/miss cases.",
    )
    parser.add_argument(
        "--stale-track-radius-km",
        type=float,
        default=500.0,
        help="Maximum radius for stale cyclone/remnant association. Default: 500.",
    )
    parser.add_argument(
        "--same-system-radius-km",
        type=float,
        default=700.0,
        help="Suppress duplicate hotspots tied to the same storm. Default: 700.",
    )
    parser.add_argument(
        "--min-latitude",
        type=float,
        default=-35.0,
    )
    parser.add_argument(
        "--max-latitude",
        type=float,
        default=35.0,
    )
    parser.add_argument(
        "--maximum-land-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--minimum-surface-temperature-k",
        type=float,
        default=None,
        help=(
            "Optional surface-temperature pre-filter. Disabled by default "
            "because it must match the training methodology."
        ),
    )
    parser.add_argument(
        "--poll-interval-hours",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=0.75,
        help="Neural detection threshold for all selected horizons. Default: 0.75",
    )
    parser.add_argument(
        "--threshold-file",
        type=Path,
        default=MODELS_DIR / "operational_thresholds.json",
    )
    parser.add_argument(
        "--training-metadata-file",
        type=Path,
        default=MODELS_DIR / "training_metadata.json",
    )
    parser.add_argument(
        "--channel-names-file",
        type=Path,
        default=MODELS_DIR / "channel_names.json",
    )
    parser.add_argument(
        "--means-file",
        type=Path,
        default=MODELS_DIR / "channel_means.npy",
    )
    parser.add_argument(
        "--stds-file",
        type=Path,
        default=MODELS_DIR / "channel_stds.npy",
    )

    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not 0.0 < args.alert_threshold < 1.0:
        raise ValueError("--alert-threshold must be between 0 and 1.")
    if args.nms_radius_km <= 0:
        raise ValueError("--nms-radius-km must be positive.")
    if not 0.0 <= args.min_expert_confidence < 1.0:
        raise ValueError("--min-expert-confidence must be in [0,1).")
    if args.max_detections < 1:
        raise ValueError("--max-detections must be at least 1.")
    if args.stale_track_radius_km <= 0 or args.same_system_radius_km <= 0:
        raise ValueError("Track and same-system radii must be positive.")
    if args.min_latitude >= args.max_latitude:
        raise ValueError("--min-latitude must be lower than --max-latitude.")
    if not 0.0 <= args.maximum_land_fraction <= 1.0:
        raise ValueError("--maximum-land-fraction must be between 0 and 1.")
    if args.poll_interval_hours < 1:
        raise ValueError("--poll-interval-hours must be at least 1.")


def make_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        allow_domain_shift=args.allow_domain_shift,
        insecure_ssl=args.insecure_ssl,
        threshold_file=args.threshold_file.resolve(),
        alert_threshold=args.alert_threshold,
        training_metadata_file=args.training_metadata_file.resolve(),
        channel_names_file=args.channel_names_file.resolve(),
        means_file=args.means_file.resolve(),
        stds_file=args.stds_file.resolve(),
        batch_size=args.batch_size,
        nms_radius_km=args.nms_radius_km,
        min_expert_confidence=args.min_expert_confidence,
        max_detections=args.max_detections,
        active_systems_json=(
            args.active_systems_json.resolve()
            if args.active_systems_json is not None
            else None
        ),
        classification_mode=args.classification_mode,
        post_classifier_model=(
            args.post_classifier_model.resolve()
            if args.post_classifier_model is not None
            else None
        ),
        stale_track_radius_km=args.stale_track_radius_km,
        same_system_radius_km=args.same_system_radius_km,
        min_latitude=args.min_latitude,
        max_latitude=args.max_latitude,
        maximum_land_fraction=args.maximum_land_fraction,
        minimum_surface_temperature_k=args.minimum_surface_temperature_k,
        poll_interval_hours=args.poll_interval_hours,
        lead_times=args.lead_times,
    )


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    validate_arguments(args)
    config = make_runtime_config(args)

    if args.once:
        execute_pipeline_cycle(config)
        return

    while True:
        try:
            execute_pipeline_cycle(config)
        except KeyboardInterrupt:
            logging.info("Pipeline stopped by user.")
            return
        except Exception:
            logging.exception("Operational cycle failed.")

        logging.info(
            "Sleeping for %d hour(s) before the next cycle.",
            config.poll_interval_hours,
        )
        time.sleep(config.poll_interval_hours * 3600)


if __name__ == "__main__":
    main()
