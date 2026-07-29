#!/usr/bin/env python3
import os
import glob
import json
import logging
import numpy as np
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

FEATURES_DIR = "/workspace/data/extracted_features_entire_world"
OUTPUT_STATS_PATH = "/workspace/scaling_stats.json"

V3D_NAMES = ['ugrdprs', 'vgrdprs', 'vvelprs', 'tmpprs', 'rhprs', 'hgtprs', 'absvprs']
V2D_NAMES = ['pressfc', 'capesfc', 'tmpsfc', 'landmask']

def compute_exact_dataset_stats():
    nc_files = sorted(glob.glob(os.path.join(FEATURES_DIR, "*.nc")))
    
    if not nc_files:
        logging.critical(f"No feature files found in {FEATURES_DIR}!")
        return

    # Subsample ~400 files uniformly across the dataset for fast & representative stats
    sample_stride = max(1, len(nc_files) // 400)
    sampled_files = nc_files[::sample_stride]

    logging.info(f"Found {len(nc_files)} total files. Computing exact mathematical profile using {len(sampled_files)} representative samples...")

    stats_accumulator = {}
    all_vars = V3D_NAMES + V2D_NAMES
    
    for v in all_vars:
        stats_accumulator[v] = {"sum": 0.0, "sq_sum": 0.0, "count": 0}

    for idx, fpath in enumerate(sampled_files):
        if idx % 50 == 0:
            logging.info(f"Processing sample {idx}/{len(sampled_files)}: {os.path.basename(fpath)}")
            
        try:
            # Disable xarray auto-decoding to avoid 'dtype in attrs' metadata conflicts
            ds = xr.open_dataset(
                fpath, 
                mask_and_scale=False, 
                decode_times=False
            )
            
            for var_name in all_vars:
                if var_name not in ds:
                    continue
                
                arr = ds[var_name].values.astype(np.float64)
                valid_mask = ~np.isnan(arr)
                valid_vals = arr[valid_mask]
                
                if valid_vals.size == 0:
                    continue
                
                stats_accumulator[var_name]["sum"] += np.sum(valid_vals)
                stats_accumulator[var_name]["sq_sum"] += np.sum(valid_vals ** 2)
                stats_accumulator[var_name]["count"] += valid_vals.size
                
            ds.close()
        except Exception as e:
            logging.warning(f"Error reading {fpath}: {e}")

    exact_stats = {}
    for var_name, acc in stats_accumulator.items():
        if acc["count"] == 0:
            logging.error(f"No valid data found for variable: {var_name}")
            continue
            
        mean_val = acc["sum"] / acc["count"]
        var_val = (acc["sq_sum"] / acc["count"]) - (mean_val ** 2)
        std_val = np.sqrt(max(var_val, 1e-8))
        
        exact_stats[var_name] = {
            "mean": float(mean_val),
            "std": float(std_val)
        }
        
        logging.info(f"[{var_name}] Mean: {mean_val:.6f} | Std: {std_val:.6f}")

    with open(OUTPUT_STATS_PATH, "w") as f:
        json.dump(exact_stats, f, indent=4)

    logging.info(f"✅ Successfully exported mathematical scaling profile to {OUTPUT_STATS_PATH}")

if __name__ == "__main__":
    compute_exact_dataset_stats()