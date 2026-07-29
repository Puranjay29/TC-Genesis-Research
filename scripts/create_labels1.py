#!/usr/bin/env python3

"""
This script generates ground-truth label metadata pairing ERA5/reanalysis NetCDF files 
with IBTrACS tropical cyclone tracks for Tropical Cyclogenesis (TCG) prediction.

Outputs 'tc_{leadtime}h.csv' containing:
  - Genesis status and exact genesis coordinates [Lat, Lon]
  - All active TC positions at the observation time (for exclusion masking)
  - File paths and lead-time matched timestamps
"""

import argparse
from datetime import datetime, timedelta
import glob
import json
import os
import pandas as pd
from tqdm import tqdm
import xarray as xr


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description="Generate TCG Labels from IBTrACS and ERA5 dataset.")

    parser.add_argument(
        '--best-track',
        dest='best_track',
        action='store',
        required=True,
        help='Path to IBTrACS CSV file (e.g., ibtracs.ALL.list.v04r01.csv).')

    parser.add_argument(
        '--observations-dir',
        dest='observations_dir',
        action='store',
        required=True,
        help='''
            Path to directory containing observation .nc files.
            The output file will be saved as `tc_{leadtime}h.csv` in this folder.
            ''')

    parser.add_argument(
        '--leadtime',
        dest='leadtime',
        default=24,
        type=int,
        help='Forecast lead time in hours (default: 24).')

    return parser.parse_args(args)


def parse_date_from_nc_filename(filepath: str) -> datetime:
    """Extracts timestamp from NetCDF filename assuming standard format *_YYYYMMDD_HH_MM.nc"""
    filename = os.path.basename(filepath)
    filename_no_ext, _ = os.path.splitext(filename)
    parts = filename_no_ext.split('_')
    
    # Handle patterns like fnl_20080501_00_00.nc or features_20080501_00_00.nc
    date_str = f"{parts[-3]}_{parts[-2]}_{parts[-1]}"
    return datetime.strptime(date_str, '%Y%m%d_%H_%M')


def list_reanalysis_files(path: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(path, '*.nc')))
    if not files:
        raise ValueError(f"No NetCDF (.nc) files found in path: {path}")
    
    parsed_files = []
    for f in files:
        try:
            parsed_files.append((parse_date_from_nc_filename(f), f))
        except Exception as e:
            print(f"Warning: Skipping file {f} due to date parsing error: {e}")
            continue
            
    if not parsed_files:
        raise ValueError(f"Could not parse valid dates from any NetCDF files in: {path}")

    dates, filepaths = zip(*parsed_files)
    return pd.DataFrame({
        'OriginalDate': dates,
        'Path': filepaths
    })


def load_best_track(
        path: str,
        domain: tuple[float, float, float, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and standardizes IBTrACS data within geographical domain bounds."""
    latmin, latmax, lonmin, lonmax = domain

    # Read IBTrACS dataset, skipping line 1 (units row)
    df = pd.read_csv(path, skiprows=[1], na_filter=False, low_memory=False)
    
    # Ensure numerical conversion for spatial coordinates
    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
    df = df.dropna(subset=['LAT', 'LON'])

    # Parse ISO_TIME cleanly
    df['Date'] = pd.to_datetime(df['ISO_TIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['Date'])

    # Standardize Longitudes to [0.0, 360.0) range to match ERA5 global grids
    df['LON'] = df['LON'].apply(lambda l: l % 360.0)

    # Filter by domain boundaries
    lon_mask = (df['LON'] >= lonmin) & (df['LON'] <= lonmax)
    lat_mask = (df['LAT'] >= latmin) & (df['LAT'] <= latmax)
    df_domain = df[lon_mask & lat_mask].copy()

    # Identify genesis records: first recorded point for each unique Storm ID (SID)
    genesis_df = df_domain.groupby('SID', sort=False).first().reset_index()

    return genesis_df, df_domain


def create_labels(files_df: pd.DataFrame, best_track_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairs observation dates with IBTrACS genesis events and active tropical cyclone tracks.
    """
    rows = []

    for _, row in tqdm(files_df.iterrows(), total=len(files_df), desc="Generating Labels"):
        obs_path = row['Path']
        obs_time = row['OriginalDate']
        target_genesis_time = row['TargetGenesisDate']

        # 1. Find all genesis events occurring at target_genesis_time (OriginalDate + leadtime)
        genesis_matches = best_track_df[best_track_df['Date'] == target_genesis_time]
        
        genesis_events = []
        if len(genesis_matches) > 0:
            for sid, group in genesis_matches.groupby('SID'):
                full_storm_track = best_track_df[best_track_df['SID'] == sid]
                first_obs_time = full_storm_track['Date'].min()
                
                if target_genesis_time == first_obs_time:
                    first_row = group.iloc[0]
                    genesis_events.append({
                        'SID': sid,
                        'LAT': float(first_row['LAT']),
                        'LON': float(first_row['LON']),
                        'NAME': str(first_row.get('NAME', 'UNNAMED'))
                    })

        is_genesis = len(genesis_events) > 0

        # 2. Find ALL active TCs existing at the observation time (obs_time)
        active_tcs_at_obs = best_track_df[best_track_df['Date'] == obs_time]
        active_tc_locations = [
            [float(lat), float(lon)] 
            for lat, lon in zip(active_tcs_at_obs['LAT'], active_tcs_at_obs['LON'])
        ]
        active_tc_sids = active_tcs_at_obs['SID'].tolist()

        if is_genesis:
            genesis_lats = [g['LAT'] for g in genesis_events]
            genesis_lons = [g['LON'] for g in genesis_events]
            genesis_sids = [g['SID'] for g in genesis_events]

            label_entry = {
                'Observation_Date': obs_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Target_Genesis_Date': target_genesis_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Genesis': True,
                'Genesis_Count': len(genesis_events),
                'Genesis_SIDs': json.dumps(genesis_sids),
                'Genesis_Lats': json.dumps(genesis_lats),
                'Genesis_Lons': json.dumps(genesis_lons),
                'Latitude': genesis_lats[0],
                'Longitude': genesis_lons[0],
                'Is_Active_TC_Present': len(active_tc_locations) > 0,
                'Active_TC_Locations': json.dumps(active_tc_locations),
                'Active_TC_SIDs': json.dumps(active_tc_sids),
                'Path': obs_path
            }
        else:
            label_entry = {
                'Observation_Date': obs_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Target_Genesis_Date': target_genesis_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Genesis': False,
                'Genesis_Count': 0,
                'Genesis_SIDs': json.dumps([]),
                'Genesis_Lats': json.dumps([]),
                'Genesis_Lons': json.dumps([]),
                'Latitude': None,
                'Longitude': None,
                'Is_Active_TC_Present': len(active_tc_locations) > 0,
                'Active_TC_Locations': json.dumps(active_tc_locations),
                'Active_TC_SIDs': json.dumps(active_tc_sids),
                'Path': obs_path
            }

        rows.append(label_entry)

    return pd.DataFrame(rows)


def get_domain(path: str) -> tuple[float, float, float, float]:
    """Reads latitude and longitude bounds directly from a NetCDF reanalysis file."""
    with xr.open_dataset(path, decode_cf=False) as ds:
        lat_key = 'lat' if 'lat' in ds else 'latitude'
        lon_key = 'lon' if 'lon' in ds else 'longitude'
        
        lat = ds[lat_key].values
        lon = ds[lon_key].values
        
        lon_min = float(lon.min()) % 360.0
        lon_max = float(lon.max()) % 360.0
        if lon_max < lon_min:
            lon_min, lon_max = 0.0, 360.0

        return float(lat.min()), float(lat.max()), lon_min, lon_max


def main(args=None):
    args = parse_arguments(args)

    print(f"--> Listing reanalysis NetCDF files from: {args.observations_dir}")
    files_df = list_reanalysis_files(args.observations_dir)
    
    files_df['TargetGenesisDate'] = files_df['OriginalDate'].apply(
        lambda d: d + timedelta(hours=args.leadtime)
    )
    files_df = files_df.sort_values('OriginalDate').reset_index(drop=True)

    print(f"--> Extracted {len(files_df)} NetCDF observation timestamps.")
    
    domain = get_domain(files_df['Path'].iloc[0])
    print(f"--> Dataset Domain Lat Bounds: [{domain[0]:.1f}°, {domain[1]:.1f}°], "
          f"Lon Bounds: [{domain[2]:.1f}°, {domain[3]:.1f}°]")

    print(f"--> Loading IBTrACS best-track dataset from: {args.best_track}")
    _, best_track_df = load_best_track(args.best_track, domain)

    labels_df = create_labels(files_df, best_track_df)
    
    output_path = os.path.join(args.observations_dir, f'tc_{args.leadtime}h.csv')
    labels_df.to_csv(output_path, index=False)
    
    pos_count = labels_df['Genesis'].sum()
    total_count = len(labels_df)
    print(f"\nLabel generation complete!")
    print(f"  • Total Observation Timestamps : {total_count}")
    print(f"  • Positive Genesis Timestamps  : {pos_count}")
    print(f"  • Output Saved To              : {output_path}")


if __name__ == '__main__':
    main()