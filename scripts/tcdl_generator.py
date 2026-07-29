#!/usr/bin/env python3

"""
Production-Grade Process-Parallel Patch Generator for Operational TC Genesis Detection.

Architecture & Robustness:
  - ProcessPoolExecutor parallel extraction (zero C-level HDF5 lock deadlocks).
  - Fixed precomputed tropical normalization (channel_means.npy / channel_stds.npy).
  - Hard-negative disturbance sampling (70% biased toward vorticity waves).
  - Expanded Haversine exclusion radius (550 km) protecting pre-genesis bands.
  - Normalized-space meteorological augmentations (untouched landmask).
  - Deterministic validation set (random_state=42).
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Computes Great-Circle (Haversine) distance between coordinates in kilometers."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians((lon2 - lon1 + 180.0) % 360.0 - 180.0)
    
    a = (np.sin(dlat / 2.0) ** 2 + 
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def _standalone_extract_patch(file_path, center_lat, center_lon, half_patch, patch_size, lat_offset=0, lon_offset=0):
    """Standalone worker function for process-safe NetCDF extraction."""
    if not os.path.exists(file_path):
        return None

    v3d = ['ugrdprs', 'vgrdprs', 'vvelprs', 'tmpprs', 'rhprs', 'hgtprs', 'absvprs']
    v2d = ['pressfc', 'capesfc', 'tmpsfc', 'landmask']

    try:
        with xr.open_dataset(file_path, decode_cf=False, decode_times=False) as ds_raw:
            for var in ds_raw.variables:
                if 'dtype' in ds_raw[var].attrs:
                    del ds_raw[var].attrs['dtype']
            ds = xr.decode_cf(ds_raw, decode_timedelta=False)

            lat_arr = ds.lat.values
            lon_arr = ds.lon.values

            lat_idx = np.abs(lat_arr - center_lat).argmin() + lat_offset
            lon_idx = np.abs(lon_arr - center_lon).argmin() + lon_offset

            lat_idx = max(half_patch, min(len(lat_arr) - half_patch - 1, lat_idx))
            lon_idx = max(half_patch, min(len(lon_arr) - half_patch - 1, lon_idx))

            lat_start = lat_idx - half_patch
            lat_end = lat_start + patch_size

            lon_start = lon_idx - half_patch
            lon_end = lon_start + patch_size

            tensor_channels = []

            for var in v3d:
                data_3d = ds[var].values[:, lat_start:lat_end, lon_start:lon_end]
                for level_idx in range(data_3d.shape[0]):
                    tensor_channels.append(data_3d[level_idx])

            for var in v2d:
                data_2d = ds[var].values[lat_start:lat_end, lon_start:lon_end]
                tensor_channels.append(data_2d)

            sample_array = np.stack(tensor_channels, axis=-1).astype(np.float32)
            sample_array = np.nan_to_num(sample_array, nan=0.0, posinf=0.0, neginf=0.0)
            return sample_array

    except Exception:
        return None


def _worker_extract_pos(args_tuple):
    """Worker task to extract a single positive patch."""
    file_path, target_lat, target_lon, half_patch, patch_size, enable_augmentation = args_tuple
    lat_off = np.random.randint(-2, 3) if (enable_augmentation and np.random.rand() < 0.7) else 0
    lon_off = np.random.randint(-2, 3) if (enable_augmentation and np.random.rand() < 0.7) else 0
    return _standalone_extract_patch(file_path, target_lat, target_lon, half_patch, patch_size, lat_off, lon_off)


def _worker_extract_neg(args_tuple):
    """Worker task to search for and extract a valid hard-negative patch."""
    row_dict, half_patch, patch_size, exclusion_radius_km, enable_augmentation = args_tuple
    file_path = row_dict['Path']

    if not os.path.exists(file_path):
        return None

    exclusion_coords = []
    if 'Active_TC_Locations' in row_dict and pd.notna(row_dict['Active_TC_Locations']):
        try:
            active_locs = json.loads(row_dict['Active_TC_Locations'])
            exclusion_coords.extend(active_locs)
        except Exception:
            pass

    if row_dict['Genesis']:
        if 'Genesis_Lats' in row_dict and pd.notna(row_dict['Genesis_Lats']):
            try:
                g_lats = json.loads(row_dict['Genesis_Lats'])
                g_lons = json.loads(row_dict['Genesis_Lons'])
                for glat, glon in zip(g_lats, g_lons):
                    exclusion_coords.append([glat, glon])
            except Exception:
                exclusion_coords.append([row_dict['Latitude'], row_dict['Longitude']])
        else:
            exclusion_coords.append([row_dict['Latitude'], row_dict['Longitude']])

    try:
        with xr.open_dataset(file_path, decode_cf=False, decode_times=False) as ds_raw:
            for var in ds_raw.variables:
                if 'dtype' in ds_raw[var].attrs:
                    del ds_raw[var].attrs['dtype']
            ds = xr.decode_cf(ds_raw, decode_timedelta=False)

            lat_arr = ds.lat.values
            lon_arr = ds.lon.values
            landmask = ds['landmask'].values
            vorticity = ds['absvprs'].values[0]

            trop_lat_idxs = np.where((lat_arr >= -35.0) & (lat_arr <= 35.0))[0]
            if len(trop_lat_idxs) == 0:
                return None

            prefer_hard_negative = np.random.rand() < 0.70

            for _ in range(15):
                rand_lat_i = np.random.choice(trop_lat_idxs)
                rand_lon_i = np.random.randint(half_patch, len(lon_arr) - half_patch)

                cand_lat = float(lat_arr[rand_lat_i])
                cand_lon = float(lon_arr[rand_lon_i]) % 360.0

                if prefer_hard_negative and abs(vorticity[rand_lat_i, rand_lon_i]) < 1e-5:
                    continue

                patch_land = landmask[
                    max(0, rand_lat_i - half_patch):rand_lat_i + half_patch + 1,
                    max(0, rand_lon_i - half_patch):rand_lon_i + half_patch + 1
                ]
                if np.mean(patch_land) > 0.3:
                    continue

                is_excluded = False
                for ex_lat, ex_lon in exclusion_coords:
                    dist = haversine_distance(cand_lat, cand_lon, float(ex_lat), float(ex_lon) % 360.0)
                    if dist < exclusion_radius_km:
                        is_excluded = True
                        break

                if not is_excluded:
                    lat_off = np.random.randint(-1, 2) if (enable_augmentation and np.random.rand() < 0.5) else 0
                    lon_off = np.random.randint(-1, 2) if (enable_augmentation and np.random.rand() < 0.5) else 0

                    return _standalone_extract_patch(file_path, cand_lat, cand_lon, half_patch, patch_size, lat_off, lon_off)

    except Exception:
        pass

    return None


class TropicalCyclogenesisGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        csv_file,
        batch_size=32,
        patch_size=21,
        neg_to_pos_ratio=3,
        exclusion_radius_km=550.0,
        is_train=True,
        enable_augmentation=True,
        stats_dir='saved_models',
        num_workers=8
    ):
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.exclusion_radius_km = exclusion_radius_km
        self.is_train = is_train
        self.enable_augmentation = is_train and enable_augmentation
        self.num_workers = num_workers

        self.df = pd.read_csv(csv_file)
        date_col = 'Observation_Date' if 'Observation_Date' in self.df.columns else 'Date'
        self.df['Date'] = pd.to_datetime(self.df[date_col])

        means_path = os.path.join(stats_dir, 'channel_means.npy')
        stds_path = os.path.join(stats_dir, 'channel_stds.npy')

        if os.path.exists(means_path) and os.path.exists(stds_path):
            self.channel_means = np.load(means_path)
            self.channel_stds = np.load(stds_path)
            print(f"--> Loaded Fixed Global Channel Means and Stds from: {stats_dir}")
        else:
            print(f"--> Warning: Fixed stats not found in '{stats_dir}'. Falling back to online estimation...")
            self.channel_means = None
            self.channel_stds = None

        self.pos_df = self.df[self.df['Genesis'] == True].reset_index(drop=True)
        split_type = "TRAIN" if self.is_train else "VALIDATION/TEST"
        print(f"--> Initializing Generator [{split_type}]: {os.path.basename(csv_file)}")
        print(f"    Found {len(self.pos_df)} Positive Timesteps | Worker Processes: {self.num_workers}")

        if not self.is_train:
            self._build_fixed_validation_dataset()
        else:
            self.resample_dataset()

    def _normalize_patch(self, patch):
        """Performs per-channel z-score normalization for physical channels (0..135)."""
        norm_patch = patch.copy()
        if self.channel_means is not None and self.channel_stds is not None:
            norm_patch[:, :, :136] = (norm_patch[:, :, :136] - self.channel_means) / self.channel_stds
        return norm_patch

    def _sample_positive_patches(self):
        """Extracts positive patches in parallel using ProcessPoolExecutor."""
        pos_patches = []
        desc_str = "Parallel Positives (Train)" if self.is_train else "Parallel Positives (Val)"

        task_args = [
            (
                row['Path'],
                float(row['Latitude']),
                float(row['Longitude']) % 360.0,
                self.half_patch,
                self.patch_size,
                self.enable_augmentation
            )
            for _, row in self.pos_df.iterrows()
        ]

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(_worker_extract_pos, arg) for arg in task_args]
            for future in tqdm(as_completed(futures), total=len(task_args), desc=desc_str, leave=False):
                res = future.result()
                if res is not None:
                    pos_patches.append(res)

        return np.array(pos_patches, dtype=np.float32)

    def _sample_dynamic_negatives(self, count_needed):
        """Extracts dynamic hard-negatives in parallel using ProcessPoolExecutor."""
        negative_patches = []
        
        if not self.is_train:
            df_shuffled = self.df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        else:
            df_shuffled = self.df.sample(frac=1.0).reset_index(drop=True)

        desc_str = "Parallel Negatives (Train)" if self.is_train else "Parallel Negatives (Val)"
        
        row_idx = 0
        total_rows = len(df_shuffled)

        with tqdm(total=count_needed, desc=desc_str, leave=True) as pbar:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                while len(negative_patches) < count_needed:
                    batch_dicts = [
                        df_shuffled.iloc[(row_idx + i) % total_rows].to_dict()
                        for i in range(self.num_workers * 2)
                    ]
                    row_idx += len(batch_dicts)

                    task_args = [
                        (
                            rd,
                            self.half_patch,
                            self.patch_size,
                            self.exclusion_radius_km,
                            self.enable_augmentation
                        )
                        for rd in batch_dicts
                    ]

                    futures = [executor.submit(_worker_extract_neg, arg) for arg in task_args]
                    for future in as_completed(futures):
                        res = future.result()
                        if res is not None and len(negative_patches) < count_needed:
                            negative_patches.append(res)
                            pbar.update(1)

        return np.array(negative_patches, dtype=np.float32)

    def _apply_augmentations_normalized_space(self, norm_patch):
        """Applies field noise and perturbations in standard normalized space."""
        aug_patch = norm_patch.copy()
        
        met_channels = aug_patch[:, :, :136]
        landmask_channel = aug_patch[:, :, 136:]

        if np.random.rand() < 0.50:
            noise = np.random.normal(loc=0.0, scale=0.01, size=met_channels.shape)
            met_channels += noise.astype(np.float32)

        if np.random.rand() < 0.30:
            scales = np.random.uniform(0.99, 1.01, size=(1, 1, met_channels.shape[-1]))
            met_channels *= scales.astype(np.float32)

        if np.random.rand() < 0.20:
            mask = np.random.rand(*met_channels.shape[:2], 1) > 0.01
            met_channels *= mask.astype(np.float32)

        return np.concatenate([met_channels, landmask_channel], axis=-1)

    def _build_fixed_validation_dataset(self):
        """Constructs a deterministic, fixed validation/test evaluation set."""
        print("--> Constructing Fixed Validation/Test evaluation set...")
        pos_samples = self._sample_positive_patches()
        neg_samples = self._sample_dynamic_negatives(len(pos_samples))

        X_data = []
        Y_data = []

        for p in pos_samples:
            X_data.append(self._normalize_patch(p))
            Y_data.append(1.0)

        for n in neg_samples:
            X_data.append(self._normalize_patch(n))
            Y_data.append(0.0)

        self.samples_x = np.array(X_data, dtype=np.float32)
        self.samples_y = np.array(Y_data, dtype=np.float32)

    def resample_dataset(self):
        """Resamples dynamic training positives and ocean negatives every epoch."""
        pos_samples = self._sample_positive_patches()
        num_pos = len(pos_samples)
        num_neg = num_pos * self.neg_to_pos_ratio

        neg_samples = self._sample_dynamic_negatives(num_neg)

        X_data = []
        Y_data = []

        for p in pos_samples:
            norm_p = self._normalize_patch(p)
            sample = self._apply_augmentations_normalized_space(norm_p) if self.enable_augmentation else norm_p
            X_data.append(sample)
            Y_data.append(1.0)

        for n in neg_samples:
            norm_n = self._normalize_patch(n)
            sample = self._apply_augmentations_normalized_space(norm_n) if self.enable_augmentation else norm_n
            X_data.append(sample)
            Y_data.append(0.0)

        X_data = np.array(X_data, dtype=np.float32)
        Y_data = np.array(Y_data, dtype=np.float32)

        indices = np.random.permutation(len(X_data))
        self.samples_x = X_data[indices]
        self.samples_y = Y_data[indices]

    def on_epoch_end(self):
        """Refreshes positive offsets and ocean negatives at the end of each training epoch."""
        if self.is_train:
            self.resample_dataset()

    def __len__(self):
        return int(np.ceil(len(self.samples_x) / self.batch_size))

    def __getitem__(self, index):
        batch_x = self.samples_x[index * self.batch_size : (index + 1) * self.batch_size]
        batch_y = self.samples_y[index * self.batch_size : (index + 1) * self.batch_size]
        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)