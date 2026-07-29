#!/usr/bin/env python3

"""
Logging Script for Out-of-Sample Test Predictions.
Extracts complete spatial metadata (Latitude/Longitude/TC_ID) directly 
from generator internal positive and negative metadata tables.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from tcdl_generator import TropicalCyclogenesisGenerator
from train_vit_gru import PatchExtractorAndEmbedder, SqueezeAndExcitationBlock
from train_ensemble import LayerScale, build_advanced_ensemble

tf.config.optimizer.set_experimental_options({"layout_optimizer": False})


def parse_arguments():
    parser = argparse.ArgumentParser(description="Log Test Split Predictions for TC Genesis Models.")
    parser.add_argument('--leadtime', type=int, default=24, choices=[24, 48, 72])
    parser.add_argument('--batch-size', type=int, default=36)
    parser.add_argument('--stats-dir', type=str, default='saved_models')
    return parser.parse_args()


def load_model_safely(model_path, custom_objects):
    try:
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False, safe_mode=False)
    except Exception as e:
        print(f"⚠️ Standard load_model failed for {model_path}: {e}")
        return None


def get_val_from_meta(meta_row, candidate_keys):
    """Safely extracts coordinate or ID from candidate keys."""
    for key in candidate_keys:
        if key in meta_row and pd.notna(meta_row[key]):
            return meta_row[key]
    return np.nan


def main():
    args = parse_arguments()

    test_csv = f"data/extracted_features_entire_world/tc_{args.leadtime}h_test.csv"
    output_log_path = f"data/extracted_features_entire_world/test_predictions_log_{args.leadtime}h.csv"

    cnn_path = f'saved_models/best_cnn_lstm_{args.leadtime}h.keras'
    vit_path = f'saved_models/best_vit_gru_{args.leadtime}h.keras'
    ens_path = f'saved_models/best_ensemble_{args.leadtime}h.keras'

    print(f"\n=================== LOGGING TEST PREDICTIONS ({args.leadtime}h LEAD TIME) ===================")
    print(f"  • Test Dataset : {test_csv}")
    print(f"  • Output Log   : {output_log_path}")

    # Step 1: Initialize Data Generator
    print("\n--- STEP 1: INITIALIZING TEST GENERATOR ---")
    test_gen = TropicalCyclogenesisGenerator(
        test_csv,
        batch_size=args.batch_size,
        patch_size=21,
        is_train=False,
        enable_augmentation=False,
        stats_dir=args.stats_dir
    )

    y_true = test_gen.samples_y
    custom_objects = {
        'SqueezeAndExcitationBlock': SqueezeAndExcitationBlock,
        'PatchExtractorAndEmbedder': PatchExtractorAndEmbedder,
        'LayerScale': LayerScale
    }

    # Step 2: Load Backbones & Ensemble
    print("\n--- STEP 2: LOADING MODEL CHECKPOINTS ---")
    cnn_model = load_model_safely(cnn_path, custom_objects)
    vit_model = load_model_safely(vit_path, custom_objects)
    ensemble_model, _, _ = build_advanced_ensemble(cnn_path, vit_path, input_shape=(21, 21, 137))
    ensemble_model.load_weights(ens_path)

    # Step 3: Run Inference
    print("\n--- STEP 3: RUNNING BATCH INFERENCE PASSES ---")
    p_cnn = cnn_model.predict(test_gen, verbose=1).flatten()[:len(y_true)]
    p_vit = vit_model.predict(test_gen, verbose=1).flatten()[:len(y_true)]
    p_ens = ensemble_model.predict(test_gen, verbose=1).flatten()[:len(y_true)]

    # Step 4: Access Generator Meta Tables Directly
    print("\n--- STEP 4: EXTRACTING SPATIAL COORDINATES FROM GENERATOR METADATA ---")
    
    # Extract internal DataFrames from generator
    pos_df = getattr(test_gen, 'pos_df', pd.DataFrame()).reset_index(drop=True)
    neg_df = getattr(test_gen, 'neg_df', pd.DataFrame()).reset_index(drop=True)
    full_df = getattr(test_gen, 'df', pd.DataFrame()).reset_index(drop=True)

    lat_keys = ['Latitude', 'latitude', 'lat', 'Center_Lat', 'center_lat', 'lat_center']
    lon_keys = ['Longitude', 'longitude', 'lon', 'Center_Lon', 'center_lon', 'lon_center']
    id_keys  = ['TC_ID', 'tc_id', 'Storm_ID', 'storm_id', 'ID', 'id']

    num_pos = len(pos_df) if len(pos_df) > 0 else int(np.sum(y_true == 1))
    
    log_records = []

    for idx in range(len(y_true)):
        label = int(y_true[idx])
        
        prob_cnn_val = float(p_cnn[idx])
        prob_vit_val = float(p_vit[idx])
        prob_ens_val = float(p_ens[idx])
        pred_ens_class = 1 if prob_ens_val >= 0.5 else 0

        # Select metadata source row
        if label == 1:
            if idx < len(pos_df):
                meta_row = pos_df.iloc[idx]
            elif idx < len(full_df):
                meta_row = full_df.iloc[idx]
            else:
                meta_row = {}
        else:
            neg_idx = idx - num_pos
            if 0 <= neg_idx < len(neg_df):
                meta_row = neg_df.iloc[neg_idx]
            elif idx < len(full_df):
                meta_row = full_df.iloc[idx]
            else:
                meta_row = {}

        raw_lat = get_val_from_meta(meta_row, lat_keys)
        raw_lon = get_val_from_meta(meta_row, lon_keys)
        raw_id  = get_val_from_meta(meta_row, id_keys)

        if label == 1:
            tc_id_str = str(raw_id) if pd.notna(raw_id) else f"Positive TC #{idx + 1}"
            lat_str   = round(float(raw_lat), 2) if pd.notna(raw_lat) else "N/A"
            lon_str   = round(float(raw_lon), 2) if pd.notna(raw_lon) else "N/A"
        else:
            tc_id_str = "N/A (Background Ocean)"
            lat_str   = "N/A"
            lon_str   = "N/A"

        log_records.append({
            "Sample_ID": idx + 1,
            "TC_ID": tc_id_str,
            "Target_Latitude": lat_str,
            "Target_Longitude": lon_str,
            "Actual_Label": label,
            "Prob_CNN_LSTM": round(prob_cnn_val, 4),
            "Prob_ViT_BiGRU": round(prob_vit_val, 4),
            "Prob_Ensemble": round(prob_ens_val, 4),
            "Pred_Ensemble_Class": pred_ens_class,
            "Ensemble_Match": 1 if label == pred_ens_class else 0
        })

    df_log = pd.DataFrame(log_records)
    df_log.to_csv(output_log_path, index=False)
    
    valid_coords = len(df_log[(df_log['Target_Latitude'] != 'N/A') & (df_log['Target_Longitude'] != 'N/A')])
    print(f"\n✅ Prediction log updated successfully with {valid_coords} spatial points at: {output_log_path}")


if __name__ == '__main__':
    main()