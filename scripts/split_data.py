#!/usr/bin/env python3

"""
Splits master tropical cyclogenesis (TCG) label CSVs into chronological Train, 
Validation, and Test sets to prevent data leakage across climate timelines.

Splits:
  - Train      : 2008 – 2018
  - Validation : 2019 – 2020
  - Test       : 2021 – 2022
"""

import argparse
import os
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(description="Chronological split for TCG label metadata.")
    parser.add_argument(
        '--csv-path',
        dest='csv_path',
        default='data/extracted_features_entire_world/tc_24h.csv',
        help='Path to the master label CSV file.')
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        default='data/extracted_features_entire_world',
        help='Directory to save split CSVs.')
    return parser.parse_args()


def split_dataset_by_years(csv_path: str, output_dir: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Target CSV file not found: {csv_path}")

    # Read master label sheet
    df = pd.read_csv(csv_path)

    # Detect date column name (supports new 'Observation_Date' and legacy 'Date')
    date_col = 'Observation_Date' if 'Observation_Date' in df.columns else 'Date'
    if date_col not in df.columns:
        raise KeyError(f"Could not find valid date column in {csv_path}. Columns present: {list(df.columns)}")

    df['Parsed_Date'] = pd.to_datetime(df[date_col])
    df = df.sort_values('Parsed_Date').reset_index(drop=True)
    df['Year'] = df['Parsed_Date'].dt.year

    # Chronological splits
    train_df = df[(df['Year'] >= 2008) & (df['Year'] <= 2018)].copy()
    val_df   = df[(df['Year'] >= 2019) & (df['Year'] <= 2020)].copy()
    test_df  = df[(df['Year'] >= 2021) & (df['Year'] <= 2022)].copy()

    # Drop helper processing columns
    for target_df in [train_df, val_df, test_df]:
        target_df.drop(columns=['Parsed_Date', 'Year'], inplace=True, errors='ignore')

    # Determine filename prefix based on input filename (e.g., tc_24h -> tc_24h_train.csv)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    train_path = os.path.join(output_dir, f'{base_name}_train.csv')
    val_path   = os.path.join(output_dir, f'{base_name}_val.csv')
    test_path  = os.path.join(output_dir, f'{base_name}_test.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("=================== DATASET SPLIT COMPLETE ===================")
    print(f"Master File : {csv_path}")
    print(f"TRAIN      (2008-2018) : {len(train_df):5d} timesteps | Positives: {train_df['Genesis'].sum():4d} -> {train_path}")
    print(f"VALIDATION (2019-2020) : {len(val_df):5d} timesteps | Positives: {val_df['Genesis'].sum():4d} -> {val_path}")
    print(f"TEST       (2021-2022) : {len(test_df):5d} timesteps | Positives: {test_df['Genesis'].sum():4d} -> {test_path}")
    print("==============================================================")


if __name__ == '__main__':
    args = parse_arguments()
    split_dataset_by_years(args.csv_path, args.output_dir)