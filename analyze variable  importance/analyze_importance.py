#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from train_ensemble import PatchExtractorAndEmbedder

tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

def main():
    val_csv = "data/extracted_features_entire_world/tc_24h_val.csv"
    output_plot_path = "saved_models/variable_importance.png"
    
    print("--- STEP 1: INSTANTLY VERIFYING GRAPH & LOGIC STABILITY ---")
    try:
        custom_obj = {'PatchExtractorAndEmbedder': PatchExtractorAndEmbedder}
        model = tf.keras.models.load_model('saved_models/best_ensemble.keras', custom_objects=custom_obj)
        
        # Immediate dummy test pass
        dummy_input = np.random.rand(1, 21, 21, 137).astype(np.float32)
        _ = model.predict(dummy_input, verbose=0)
        print("✅ Success! Model architecture and weight links verified.")
    except Exception as e:
        print(f"\n❌ PIPELINE VERIFICATION FAILED: {e}")
        print("Aborting immediately to save your time.")
        sys.exit(1)

    print("\n--- STEP 2: LOADING VALIDATION SET INTO IN-MEMORY CACHE ---")
    from tcdl_generator import TropicalCyclogenesisGenerator
    val_gen = TropicalCyclogenesisGenerator(val_csv, batch_size=32, patch_size=21, is_train=False)
    
    # Extract all validation samples into single stacked numpy matrices for speed
    all_x = []
    all_y = []
    for i in range(len(val_gen)):
        xb, yb = val_gen[i]
        all_x.append(xb)
        all_y.append(yb)
    
    X_val = np.concatenate(all_x, axis=0)
    y_val = np.concatenate(all_y, axis=0)
    
    print(f"Computing baseline accuracy profile on {X_val.shape[0]} arrays...")
    baseline_preds = model.predict(X_val, batch_size=64, verbose=0).flatten()
    baseline_auc = roc_auc_score(y_val, baseline_preds)
    print(f"Baseline Fused Ensemble Validation AUC: {baseline_auc:.4f}")

    # Define channel index mappings based on tcdl_generator design rules
    v3d_names = ['ugrdprs', 'vgrdprs', 'vvelprs', 'tmpprs', 'rhprs', 'hgtprs', 'absvprs']
    v2d_names = ['pressfc', 'capesfc', 'tmpsfc', 'landmask']
    
    variable_groups = {}
    
    # Map 3D vertical pressure layers (each variable occupies exactly 19 channels)
    channel_idx = 0
    for var in v3d_names:
        variable_groups[var] = list(range(channel_idx, channel_idx + 19))
        channel_idx += 19
        
    # Map 2D individual surface layers
    for var in v2d_names:
        variable_groups[var] = [channel_idx]
        channel_idx += 1

    importance_results = {}

    print("\n--- STEP 3: COMPUTING PERMUTATION IMPORTANCE MATRIX ---")
    for var_name, channels in variable_groups.items():
        # Create a clean copy of the validation data
        X_permuted = X_val.copy()
        
        # Scramble the target channels across all samples to destroy their feature signals
        for ch in channels:
            # Flatten the spatial information for these channels and shuffle across samples
            flat_shape = X_permuted[:, :, :, ch].shape
            X_permuted[:, :, :, ch] = np.random.permutation(X_permuted[:, :, :, ch].flatten()).reshape(flat_shape)
            
        # Run inference on the scrambled data
        perm_preds = model.predict(X_permuted, batch_size=64, verbose=0).flatten()
        perm_auc = roc_auc_score(y_val, perm_preds)
        
        # Importance metric is the drop in AUC score
        auc_drop = baseline_auc - perm_auc
        importance_results[var_name] = max(0.0, auc_drop)
        print(f"Variable: {var_name:<10} | Resulting AUC: {perm_auc:.4f} | Drop: {auc_drop:.4f}")

    # --- STEP 4: GENERATE VARIABLE IMPORTANCE PLOT ---
    print("\n--- STEP 4: PLOTTING FEATURE RANKING REPORT ---")
    df_imp = pd.DataFrame(list(importance_results.items()), columns=['Variable', 'Importance']).sort_values(by='Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df_imp['Variable'], df_imp['Importance'], color='dodgerblue', edgecolor='black', height=0.6)
    
    plt.title('Global Variable Importance via Permutation Drop in AUC (Fused Ensemble)', fontsize=12, weight='bold', pad=15)
    plt.xlabel('Importance Score ($\Delta$ Validation AUC Drop)', fontsize=11, weight='bold')
    plt.ylabel('Meteorological Predictors', fontsize=11, weight='bold')
    plt.grid(True, linestyle=':', alpha=0.6, axis='x')
    
    # Annotate values on the charts
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.002, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                 va='center', ha='left', fontsize=10, weight='bold')
                 
    plt.xlim(0, max(df_imp['Importance']) * 1.2)
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Success! Feature ranking visualization report saved to: {output_plot_path}")

if __name__ == '__main__':
    main()
