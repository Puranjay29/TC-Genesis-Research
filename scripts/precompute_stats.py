#!/usr/bin/env python3

"""
Parallelized Precomputation of Tropical Normalization Statistics (Mean & Standard Deviation).

Speed Optimizations:
  - Multi-process parallel processing across CPU workers using ProcessPoolExecutor.
  - Zero IPC memory bottleneck (workers return light numpy arrays).
  - Restricted strictly to tropical domain (-35° to +35° latitude).
  - Full NaN-aware accumulation with per-channel valid pixel counters.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

V3D = ['ugrdprs', 'vgrdprs', 'vvelprs', 'tmpprs', 'rhprs', 'hgtprs', 'absvprs']
V2D = ['pressfc', 'capesfc', 'tmpsfc']  # Exclude binary landmask
NUM_CHANNELS = 136  # 7 3D vars x 19 levels + 3 surface vars


def process_single_file(path: str):
    """
    Worker task: Processes a single NetCDF file, restricts to tropical latitudes,
    and returns sums, squared sums, and valid counts per channel.
    """
    c_sums = np.zeros(NUM_CHANNELS, dtype=np.float64)
    c_sq_sums = np.zeros(NUM_CHANNELS, dtype=np.float64)
    c_counts = np.zeros(NUM_CHANNELS, dtype=np.float64)

    if not os.path.exists(path):
        return c_sums, c_sq_sums, c_counts

    try:
        with xr.open_dataset(path, decode_cf=False, decode_times=False) as ds_raw:
            for var in ds_raw.variables:
                if 'dtype' in ds_raw[var].attrs:
                    del ds_raw[var].attrs['dtype']
            ds = xr.decode_cf(ds_raw, decode_timedelta=False)

            lat_arr = ds.lat.values
            trop_idx = np.where((lat_arr >= -35.0) & (lat_arr <= 35.0))[0]
            if len(trop_idx) == 0:
                return c_sums, c_sq_sums, c_counts

            channel_list = []

            for var in V3D:
                data_3d = ds[var].values[:, trop_idx, :]
                for lvl in range(data_3d.shape[0]):
                    channel_list.append(data_3d[lvl])

            for var in V2D:
                data_2d = ds[var].values[trop_idx, :]
                channel_list.append(data_2d)

            for c_idx in range(NUM_CHANNELS):
                c_data = channel_list[c_idx]
                valid_mask = np.isfinite(c_data)
                valid_vals = c_data[valid_mask].astype(np.float64)
                
                cnt = valid_vals.size
                if cnt > 0:
                    c_sums[c_idx] = np.sum(valid_vals)
                    c_sq_sums[c_idx] = np.sum(valid_vals ** 2)
                    c_counts[c_idx] = cnt

    except Exception:
        pass

    return c_sums, c_sq_sums, c_counts


def main():
    parser = argparse.ArgumentParser(description="Parallelized Precompute Tropical Normalization Stats.")
    parser.add_argument('--train-csv', default='data/extracted_features_entire_world/tc_24h_train.csv')
    parser.add_argument('--output-dir', default='saved_models')
    parser.add_argument('--processes', type=int, default=max(1, multiprocessing.cpu_count() - 2))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"Training CSV not found: {args.train_csv}")

    df = pd.read_csv(args.train_csv)
    paths = df['Path'].unique()

    print(f"--> Parallelizing Statistics Computation across {len(paths)} files...")
    print(f"    Domain Restriction : Tropical Latitudes [-35.0°, +35.0°]")
    print(f"    CPU Worker Threads : {args.processes}")

    total_sums = np.zeros(NUM_CHANNELS, dtype=np.float64)
    total_sq_sums = np.zeros(NUM_CHANNELS, dtype=np.float64)
    total_counts = np.zeros(NUM_CHANNELS, dtype=np.float64)

    with ProcessPoolExecutor(max_workers=args.processes) as executor:
        futures = {executor.submit(process_single_file, path): path for path in paths}
        
        for future in tqdm(as_completed(futures), total=len(paths), desc="Parallel Computing"):
            c_sums, c_sq_sums, c_counts = future.result()
            total_sums += c_sums
            total_sq_sums += c_sq_sums
            total_counts += c_counts

    means = total_sums / np.maximum(total_counts, 1.0)
    stds = np.sqrt(np.maximum(0.0, (total_sq_sums / np.maximum(total_counts, 1.0)) - (means ** 2))) + 1e-6

    means_formatted = means.reshape(1, 1, NUM_CHANNELS).astype(np.float32)
    stds_formatted = stds.reshape(1, 1, NUM_CHANNELS).astype(np.float32)

    means_path = os.path.join(args.output_dir, 'channel_means.npy')
    stds_path = os.path.join(args.output_dir, 'channel_stds.npy')

    np.save(means_path, means_formatted)
    np.save(stds_path, stds_formatted)

    print("\n=================== PARALLEL PRECOMPUTATION COMPLETE ===================")
    print(f"  • Processed Channels : {NUM_CHANNELS}")
    print(f"  • Saved Means File   : {means_path} | Shape: {means_formatted.shape}")
    print(f"  • Saved Stds File    : {stds_path}  | Shape: {stds_formatted.shape}")
    print("=========================================================================")


if __name__ == '__main__':
    main()