#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from train_ensemble import PatchExtractorAndEmbedder

tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

def main():
    test_csv = "data/extracted_features_entire_world/tc_24h_test.csv"
    output_plot_path = "saved_models/spatial_localization_test.png"
    
    print("--- STEP 1: INSTANTLY VERIFYING GRAPH & FILE PATH INTEGRITY ---")
    try:
        custom_obj = {'PatchExtractorAndEmbedder': PatchExtractorAndEmbedder}
        model = tf.keras.models.load_model('saved_models/best_ensemble.keras', custom_objects=custom_obj)
        
        # Read a tiny piece of the sheet to ensure paths match
        test_df = pd.read_csv(test_csv)
        print(f"✅ Success! Data tracking layout confirmed. Found {len(test_df)} test records.")
    except Exception as e:
        print(f"\n❌ PIPELINE VERIFICATION FAILED: {e}")
        print("Aborting immediately to save your time.")
        sys.exit(1)

    print("\n--- STEP 2: RUNNING DATASET PRELOAD FOR POSITIVE TARGETS ---")
    from tcdl_generator import TropicalCyclogenesisGenerator
    test_gen = TropicalCyclogenesisGenerator(test_csv, batch_size=1, patch_size=21, is_train=False)
    
    # Isolate true cyclone events out of the pre-balanced generator framework
    pos_count = len(test_gen.pos_cache_x)
    print(f"Extracting spatial points for {pos_count} positive holdout test cyclones...")
    
    latitudes = []
    longitudes = []
    predictions = []
    
    # Map predictions specifically back to the true cyclone track indexing
    for idx in range(pos_count):
        # The generator places true positive caches at the beginning of its balanced evaluation arrays
        x_sample = np.expand_dims(test_gen.samples_x[idx], axis=0)
        true_label = test_gen.samples_y[idx]
        
        # We only want to map real cyclogenesis tracks for the "where" analysis
        if true_label == 1.0:
            # Reconstruct index reference to extract original coordinates from the positive dataframe
            meta_row = test_gen.pos_df.iloc[idx]
            lat = meta_row['Latitude']
            lon = meta_row['Longitude']
            
            prob = float(model.predict(x_sample, verbose=0)[0][0])
            
            latitudes.append(lat)
            longitudes.append(lon)
            predictions.append(prob)

    print("\n--- STEP 3: GENERATING GEOGRAPHICAL DISTRIBUTION SATELLITE PLOTS ---")
    plt.figure(figsize=(14, 7))
    
    # Global basemap canvas scatter setup
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5, label="Equator")
    
    # Plot successfully spotted cyclones as green, missed ones as orange/red
    sc = plt.scatter(longitudes, latitudes, c=predictions, cmap='RdYlGn', 
                     edgecolors='black', linewidths=0.7, s=80, alpha=0.85, vmin=0.0, vmax=1.0)
    
    cbar = plt.colorbar(sc, orientation='vertical', pad=0.02)
    cbar.set_label('Ensemble Genesis Probability Score ($P \geq 0.5$ Is Correct Detection)', fontsize=11, weight='bold')
    
    plt.title('Fused Ensemble Cyclone Genesis Spatial Detection Mapping (Holdout Test 2021-2022)', fontsize=13, weight='bold', pad=15)
    plt.xlabel('Longitude (°E)', fontsize=11, weight='bold')
    plt.ylabel('Latitude (°N)', fontsize=11, weight='bold')
    plt.xlim(-180, 180)
    plt.ylim(-60, 60)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    os.makedirs('saved_models', exist_ok=True)
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Success! Spatial tracking visualization saved to: {output_plot_path}")

if __name__ == '__main__':
    main()
