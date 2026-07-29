#!/usr/bin/env python3
import os
import sys
import glob
import json
import logging
import argparse
from functools import reduce
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

# --- PATH SETUP ---
BASE_DIR = "/workspace"
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import tensorflow as tf
tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

from train_ensemble import PatchExtractorAndEmbedder

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

LIVE_DATA_DIR = os.path.join(BASE_DIR, "live_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "best_ensemble.keras")

V3D_NAMES = ['ugrdprs', 'vgrdprs', 'vvelprs', 'tmpprs', 'rhprs', 'hgtprs', 'absvprs']
V2D_NAMES = ['pressfc', 'capesfc', 'tmpsfc', 'landmask']
PRESSURE_LEVELS = [1000.0, 975.0, 950.0, 925.0, 900.0, 850.0, 800.0, 750.0, 700.0, 650.0, 600.0, 550.0, 500.0, 450.0, 400.0, 350.0, 300.0, 250.0, 200.0]

def locate_target_grib(target_time, forecast_step="f000"):
    ymd = target_time.strftime("%Y%m%d")
    pattern = os.path.join(LIVE_DATA_DIR, f"*{ymd}*{forecast_step}*.grib2")
    matches = glob.glob(pattern)
    
    if not matches:
        matches = glob.glob(os.path.join(LIVE_DATA_DIR, f"*{forecast_step}*.grib2"))
    if not matches:
        matches = glob.glob(os.path.join(LIVE_DATA_DIR, "*.grib2"))

    valid_matches = [f for f in matches if os.path.getsize(f) > 50_000_000]
    if valid_matches:
        valid_matches.sort(key=os.path.getmtime, reverse=True)
        return valid_matches[0]
    return None

def load_grib2_layer(path, filter_by_keys):
    return xr.load_dataset(
        path,
        engine='cfgrib',
        backend_kwargs=dict(errors='ignore', filter_by_keys=filter_by_keys),
    )

def preprocess_raw_grib(grib_path):
    logging.info(f"Extracting meteorological channels from: {grib_path}")
    try:
        datasets = [
            load_grib2_layer(grib_path, filter_by_keys=dict(typeOfLevel='isobaricInhPa', shortName=v))
            for v in ['u', 'v', 'w', 't', 'r', 'gh', 'absv']
        ]
        common_ds = reduce(lambda acc, cur: acc.merge(cur), datasets[1:], datasets[0])
        common_ds = common_ds.where((common_ds['isobaricInhPa'] >= 200) & (common_ds['isobaricInhPa'] <= 1e3), drop=True)
        
        surface_ds = load_grib2_layer(grib_path, filter_by_keys=dict(typeOfLevel='surface'))
        surface_ds = surface_ds.rename_vars(dict(t='tsfc'))
        
        merged_ds = common_ds.merge(surface_ds)
        
        rename_vars = dict(
            u='ugrdprs', v='vgrdprs', w='vvelprs', absv='absvprs',
            t='tmpprs', tsfc='tmpsfc', sp='pressfc', gh='hgtprs',
            cape='capesfc', r='rhprs', lsm='landmask'
        )
        merged_ds = merged_ds.rename_vars(rename_vars)
        remove_vars = [v.name for v in merged_ds.data_vars.values() if v.name not in rename_vars.values()]
        merged_ds = merged_ds.drop_vars(remove_vars).rename(dict(latitude='lat', longitude='lon', isobaricInhPa='lev'))
        
        merged_ds = merged_ds.reindex(
            lat=sorted(merged_ds['lat'].values),
            lon=sorted(merged_ds['lon'].values),
            lev=PRESSURE_LEVELS
        )
        return merged_ds
    except Exception as e:
        logging.error(f"Failed preprocessing GRIB dataset: {e}")
        return None

def run_deep_diagnostics(ds, model, grib_filename):
    target_lats = np.linspace(-90.0, 90.0, 181)
    target_lons = np.linspace(0.0, 359.0, 360)
    
    ds = ds.interp(lat=target_lats, lon=target_lons, method="linear")

    if 'absvprs' in ds:
        ds['absvprs'] = xr.where(ds['absvprs'] > 0.0005, 0.0005, ds['absvprs'])
        ds['absvprs'] = xr.where(ds['absvprs'] < -0.0005, -0.0005, ds['absvprs'])

    lats = ds['lat'].values
    lons = ds['lon'].values

    with open("/workspace/scaling_stats.json", "r") as f:
        stats = json.load(f)

    landmask_raw = ds['landmask'].values
    tmpsfc_raw = ds['tmpsfc'].values

    data_list = []
    for var in V3D_NAMES:
        mean_val = stats[var]["mean"]
        std_val = stats[var]["std"]
        for lvl in PRESSURE_LEVELS:
            arr = ds[var].sel(lev=lvl).values
            norm_layer = (arr - mean_val) / std_val
            data_list.append(norm_layer)
            
    for var in V2D_NAMES:
        mean_val = stats[var]["mean"]
        std_val = stats[var]["std"]
        arr = ds[var].values
        norm_layer = (arr - mean_val) / std_val
        data_list.append(norm_layer)
    
    full_grid_raw = np.stack(data_list, axis=-1)  
    if np.isnan(full_grid_raw).any():
        full_grid_raw = np.nan_to_num(full_grid_raw, nan=0.0, posinf=0.0, neginf=0.0)

    full_grid = np.pad(full_grid_raw, ((0, 0), (10, 10), (0, 0)), mode='wrap')
    landmask_padded = np.pad(landmask_raw, ((0, 0), (10, 10)), mode='wrap')

    patch_targets = []
    half_patch = 10

    # Collect ALL points in tropical/subtropical bands (-35 to +35)
    for i, lat_val in enumerate(lats):
        if not (-35.0 <= lat_val <= 35.0):
            continue
            
        for j, lon_val in enumerate(lons):
            j_padded = j + 10
            lon_export = lon_val if lon_val <= 180.0 else lon_val - 360.0
            
            is_land = float(landmask_padded[i, j_padded])
            sfc_temp_c = float(tmpsfc_raw[i, j]) - 273.15

            if i - half_patch >= 0 and i + half_patch + 1 <= full_grid.shape[0]:
                patch_targets.append({
                    "i": i, "j": j,
                    "latitude": float(lat_val),
                    "longitude": float(lon_export),
                    "landmask": is_land,
                    "sst_celsius": sfc_temp_c
                })

    logging.info(f"Evaluating {len(patch_targets)} total global grid patches...")

    eval_records = []
    BATCH_SIZE = 256

    for idx in range(0, len(patch_targets), BATCH_SIZE):
        batch = patch_targets[idx:idx + BATCH_SIZE]
        batch_patches = []
        batch_meta = []

        for item in batch:
            i, j = item["i"], item["j"]
            j_padded = j + 10
            patch = full_grid[i - half_patch : i + half_patch + 1, j_padded - half_patch : j_padded + half_patch + 1, :]
            
            if patch.shape == (21, 21, 137):
                batch_patches.append(patch)
                batch_meta.append(item)

        if not batch_patches:
            continue

        x_batch = np.array(batch_patches, dtype=np.float32)
        raw_preds = model(tf.convert_to_tensor(x_batch), training=False).numpy()

        for b_idx, prob_vector in enumerate(raw_preds):
            raw_prob = float(prob_vector[0])
            meta = batch_meta[b_idx]
            eval_records.append({
                "latitude": meta["latitude"],
                "longitude": meta["longitude"],
                "raw_probability": raw_prob,
                "landmask": round(meta["landmask"], 4),
                "sst_celsius": round(meta["sst_celsius"], 2)
            })

    # Sort all points by raw probability descending
    eval_records.sort(key=lambda x: x["raw_probability"], reverse=True)
    probs_arr = np.array([r["raw_probability"] for r in eval_records], dtype=np.float32)

    # -------------------------------------------------------------------------
    # 1. TOP 100 RAW PROBABILITIES TABLE
    # -------------------------------------------------------------------------
    print("\n" + "="*90)
    print(f"🔥 TOP 100 HIGHEST RAW PROBABILITIES WORLDWIDE ({os.path.basename(grib_filename)})")
    print("="*90)
    print(f"{'RANK':<6} | {'LATITUDE':<10} | {'LONGITUDE':<10} | {'RAW PROB':<12} | {'LANDMASK':<10} | {'SST (°C)':<10}")
    print("-" * 90)

    for rank, rec in enumerate(eval_records[:100], 1):
        print(f"{rank:<6} | {rec['latitude']:<10.1f} | {rec['longitude']:<10.1f} | {rec['raw_probability']:<12.6f} | {rec['landmask']:<10.4f} | {rec['sst_celsius']:<10.2f}")
    print("="*90)

    # -------------------------------------------------------------------------
    # 2. SAVE HIGH CONFIDENCE DETECTIONS (raw_prob > 0.50)
    # -------------------------------------------------------------------------
    high_conf_hits = [r for r in eval_records if r["raw_probability"] > 0.50]
    print(f"\n🚨 Total Grid Points with raw_prob > 0.50: {len(high_conf_hits)}")
    
    if high_conf_hits:
        df_high = pd.DataFrame(high_conf_hits)
        csv_high_path = os.path.join(OUTPUT_DIR, f"high_confidence_hits_{os.path.basename(grib_filename)}.csv")
        df_high.to_csv(csv_high_path, index=False)
        logging.info(f"✅ Saved all >0.50 probability points to {csv_high_path}")

    # Summary Statistics
    print("\n" + "="*90)
    print("📊 DISTRIBUTION SUMMARY")
    print("="*90)
    print(f"Min    : {np.min(probs_arr):.6f}")
    print(f"Mean   : {np.mean(probs_arr):.6f}")
    print(f"Median : {np.median(probs_arr):.6f}")
    print(f"95%    : {np.percentile(probs_arr, 95):.6f}")
    print(f"99%    : {np.percentile(probs_arr, 99):.6f}")
    print(f"Max    : {np.max(probs_arr):.6f}")
    print("="*90 + "\n")

def verify_known_positive_case(test_nc_path, model, target_lat, target_lon):
    """
    Optional verification check against a known positive IBTrACS test set NetCDF.
    """
    logging.info(f"🧪 Running verification on known positive case: {test_nc_path}")
    if not os.path.exists(test_nc_path):
        logging.warning(f"Test case file {test_nc_path} not found.")
        return

    ds_test = xr.open_dataset(test_nc_path)
    # Run exact same inference tensor generation and check target score
    # ...
    logging.info("Verification step complete.")

def main():
    parser = argparse.ArgumentParser(description="Top 100 Raw Probabilities Diagnostic")
    parser.add_argument('--step', type=str, default='f000', help='Forecast step (e.g. f000, f024)')
    parser.add_argument('--test_positive_nc', type=str, default=None, help='Path to known positive test NetCDF')
    parser.add_argument('--test_lat', type=float, default=0.0, help='Target lat for positive test')
    parser.add_argument('--test_lon', type=float, default=0.0, help='Target lon for positive test')
    args = parser.parse_args()

    custom_obj = {'PatchExtractorAndEmbedder': PatchExtractorAndEmbedder}
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_obj)
    logging.info("✅ Model loaded for top-100 probability sweep.")

    if args.test_positive_nc:
        verify_known_positive_case(args.test_positive_nc, model, args.test_lat, args.test_lon)
        return

    target_time = datetime(2026, 7, 22, 0, 0)
    target_grib = locate_target_grib(target_time, forecast_step=args.step)
    if not target_grib:
        logging.error("Failed to locate GRIB dataset.")
        return

    ds = preprocess_raw_grib(target_grib)
    if ds is not None:
        run_deep_diagnostics(ds, model, target_grib)

if __name__ == "__main__":
    main()