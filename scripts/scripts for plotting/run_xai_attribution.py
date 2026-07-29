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
    val_csv = "data/extracted_features_entire_world/tc_24h_val.csv"
    output_plot_path = "saved_models/xai_variable_importance.png"
    
    print("--- STEP 1: INSTANTLY VERIFYING XAI GRADIENT TAPE INTEGRITY ---")
    try:
        custom_obj = {'PatchExtractorAndEmbedder': PatchExtractorAndEmbedder}
        model = tf.keras.models.load_model('saved_models/best_ensemble.keras', custom_objects=custom_obj)
        
        # Verify GradientTape compatibility with a dummy variable tensor pass
        dummy_tensor = tf.convert_to_tensor(np.random.rand(1, 21, 21, 137).astype(np.float32))
        with tf.GradientTape() as tape:
            tape.watch(dummy_tensor)
            preds = model(dummy_tensor, training=False)
        _ = tape.gradient(preds, dummy_tensor)
        print("✅ Success! xAI gradient extraction engine verified and stable.")
    except Exception as e:
        print(f"\n❌ xAI GRAPH VERIFICATION FAILED: {e}")
        print("Aborting script immediately to save your time.")
        sys.exit(1)

    print("\n--- STEP 2: LOADING VALIDATION MATRIX FOR ATTRIBUTION POOL ---")
    from tcdl_generator import TropicalCyclogenesisGenerator
    val_gen = TropicalCyclogenesisGenerator(val_csv, batch_size=32, patch_size=21, is_train=False)
    
    all_x = []
    # Take a representative subset of positive cyclogenesis events for clean feature attribution
    for i in range(min(5, len(val_gen))):
        xb, yb = val_gen[i]
        # Filter for positive target tracks
        pos_idx = np.where(yb.flatten() == 1)[0]
        if len(pos_idx) > 0:
            all_x.append(xb[pos_idx])
            
    X_pos = np.concatenate(all_x, axis=0)[:50] # Benchmark on up to 50 active storm frames
    
    print(f"Extracting backpropagation gradients across {X_pos.shape[0]} cyclone cores...")
    X_tensor = tf.convert_to_tensor(X_pos.astype(np.float32))
    
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        predictions = model(X_tensor, training=False)
        
    # Backpropagate to extract feature importance scores
    gradients = tape.gradient(predictions, X_tensor)
    importances = tf.reduce_mean(tf.abs(gradients), axis=[0, 1, 2]).numpy()

    # --- STEP 3: MAPPING GRADIENTS BACK TO METEOROLOGICAL CHANNELS ---
    v3d_names = ['ugrdprs', 'vgrdprs', 'vvelprs', 'tmpprs', 'rhprs', 'hgtprs', 'absvprs']
    v2d_names = ['pressfc', 'capesfc', 'tmpsfc', 'landmask']
    
    importance_map = {}
    channel_idx = 0
    
    # Extract total attribution accumulated across all 19 pressure layers for 3D variables
    for var in v3d_names:
        var_channels = list(range(channel_idx, channel_idx + 19))
        importance_map[var] = float(np.sum(importances[var_channels]))
        channel_idx += 19
        
    for var in v2d_names:
        importance_map[var] = float(importances[channel_idx])
        channel_idx += 1

    # --- STEP 4: PLOTTING EXPLAINABLE AI ATTRIBUTION REPORT ---
    df_xai = pd.DataFrame(list(importance_map.items()), columns=['Variable', 'Saliency_Weight']).sort_values(by='Saliency_Weight', ascending=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df_xai['Variable'], df_xai['Saliency_Weight'], color='#2ecc71', edgecolor='black', height=0.6)
    plt.title('Neural Graph Variable Importance via Gradient Sensitivity (xAI Saliency Map)', fontsize=12, weight='bold', pad=15)
    plt.xlabel('Mean Absolute Gradient Magnitude ($\mathbb{E}[|\partial P / \partial X|]$)', fontsize=11, weight='bold')
    plt.ylabel('Predictor Channels', fontsize=11, weight='bold')
    plt.grid(True, linestyle=':', alpha=0.6, axis='x')
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + (max(df_xai['Saliency_Weight'])*0.01), bar.get_y() + bar.get_height()/2, f'{width:.5f}', 
                 va='center', ha='left', fontsize=10, weight='bold')
                 
    plt.xlim(0, max(df_xai['Saliency_Weight']) * 1.15)
    os.makedirs('saved_models', exist_ok=True)
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*20 + " xAI METRIC ANALYSIS COMPLETED " + "="*20)
    for index, row in df_xai.iloc[::-1].iterrows():
        print(f"Predictor: {row['Variable']:<12} | Sensitivity Weight: {row['Saliency_Weight']:.6f}")
    print("="*71)
    print(f"✅ xAI attribution plot saved directly to: {output_plot_path}")

if __name__ == '__main__':
    main()
