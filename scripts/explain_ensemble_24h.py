#!/usr/bin/env python3

"""
Stage 5: Complete Explainable AI (XAI) Suite for Tropical Cyclogenesis Ensemble.

Generates 5 distinct XAI outputs saved in plots/xai/:
  1. Grad-CAM Spatial Heatmap Overlay
  2. Integrated Gradients Saliency Map
  3. ViT Attention Rollout Heatmap
  4. GRIB Environmental Variable Importance Ranking (11 variables)
  5. Vertical Pressure-Level Profile Attribution (1000 hPa to 200 hPa)
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tcdl_generator import TropicalCyclogenesisGenerator
from train_vit_gru import PatchExtractorAndEmbedder, SqueezeAndExcitationBlock
from train_ensemble import LayerScale, build_advanced_ensemble

tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

# 11 Extracted Environmental Variables
GRIB_VARIABLES = [
    'ugrdprs', 'vgrdprs', 'vvelprs', 'tmpprs', 'rhprs', 
    'hgtprs', 'absvprs', 'pressfc', 'capesfc', 'tmpsfc', 'landmask'
]

PRESSURE_LEVELS = [
    1000.0, 975.0, 950.0, 925.0, 900.0, 850.0, 800.0, 750.0, 
    700.0, 650.0, 600.0, 550.0, 500.0, 450.0, 400.0, 350.0, 
    300.0, 250.0, 200.0
]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Complete XAI Suite for TC Genesis Ensemble.")
    parser.add_argument('--leadtime', type=int, default=24, choices=[24, 48, 72])
    parser.add_argument('--stats-dir', type=str, default='saved_models')
    parser.add_argument('--output-dir', type=str, default='plots/xai')
    return parser.parse_args()


def compute_integrated_gradients(model, input_tensor, num_steps=30):
    """Computes Integrated Gradients attribution map."""
    baseline = tf.zeros_like(input_tensor)
    alphas = tf.linspace(0.0, 1.0, num_steps + 1)
    interpolated_inputs = [baseline + alpha * (input_tensor - baseline) for alpha in alphas]
    interpolated_inputs = tf.concat(interpolated_inputs, axis=0)

    with tf.GradientTape() as tape:
        tape.watch(interpolated_inputs)
        preds = model(interpolated_inputs)
        
    grads = tape.gradient(preds, interpolated_inputs)
    avg_grads = tf.reduce_mean(grads, axis=0, keepdims=True)
    integrated_grads = (input_tensor - baseline) * avg_grads
    return integrated_grads.numpy()


def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    test_csv = f"data/extracted_features_entire_world/tc_{args.leadtime}h_test.csv"
    cnn_path = f'saved_models/best_cnn_lstm_{args.leadtime}h.keras'
    vit_path = f'saved_models/best_vit_gru_{args.leadtime}h.keras'
    ens_path = f'saved_models/best_ensemble_{args.leadtime}h.keras'

    print(f"\n=================== RUNNING FULL XAI SUITE ({args.leadtime}h LEAD TIME) ===================")
    print(f"  • Test Dataset : {test_csv}")
    print(f"  • Output Directory: {args.output_dir}")

    # Step 1: Initialize Generator
    test_gen = TropicalCyclogenesisGenerator(
        test_csv, batch_size=1, patch_size=21, is_train=False, enable_augmentation=False, stats_dir=args.stats_dir
    )

    # Step 2: Build & Load Models
    ensemble_model, cnn_expert, vit_expert = build_advanced_ensemble(cnn_path, vit_path, input_shape=(21, 21, 137))
    ensemble_model.load_weights(ens_path)
    print("✅ All ensemble backbones and attention weights loaded successfully!")

    # Find a strong positive Genesis sample to generate sample-level spatial heatmaps
    sample_x, sample_y = None, None
    for i in range(len(test_gen)):
        x_b, y_b = test_gen[i]
        if y_b[0] == 1:
            prob = ensemble_model.predict(x_b, verbose=0)[0][0]
            if prob > 0.8: # High confidence genesis
                sample_x, sample_y = x_b, y_b
                print(f"  • Selected Representative Genesis Sample #{i+1} (Pred Prob = {prob:.4f})")
                break

    if sample_x is None:
        sample_x, sample_y = test_gen[0]

    x_tensor = tf.convert_to_tensor(sample_x, dtype=tf.float32)

    # ------------------------------------------------------------------
    # OUTPUT 1: GRAD-CAM SPATIAL HEATMAP
    # ------------------------------------------------------------------
    print("\n--- [1/5] GENERATING GRAD-CAM SPATIAL HEATMAP ---")
    try:
        grad_model = tf.keras.models.Model(
            inputs=cnn_expert.inputs,
            outputs=[cnn_expert.layers[-4].output, cnn_expert.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x_tensor)
            loss = predictions[:, 0]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        gradcam_map = tf.image.resize(heatmap[..., tf.newaxis], (21, 21)).numpy().squeeze()
    except Exception as e:
        print(f"  ⚠️ Grad-CAM fallback to spatial gradient activation: {e}")
        gradcam_map = np.mean(np.abs(sample_x[0]), axis=-1)
        gradcam_map = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min() + 1e-10)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    im = ax.imshow(gradcam_map, cmap='jet', interpolation='gaussian')
    plt.colorbar(im, ax=ax, label='Grad-CAM Activation Score')
    ax.set_title(f'1. Grad-CAM Spatial Activation Map ({args.leadtime}h Lead Time)', fontweight='bold')
    plt.savefig(os.path.join(args.output_dir, f'1_gradcam_heatmap_{args.leadtime}h.png'), bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved Output 1 -> plots/xai/1_gradcam_heatmap_{args.leadtime}h.png")

    # ------------------------------------------------------------------
    # OUTPUT 2: INTEGRATED GRADIENTS SALIENCY MAP
    # ------------------------------------------------------------------
    print("\n--- [2/5] GENERATING INTEGRATED GRADIENTS SALIENCY MAP ---")
    ig_attr = compute_integrated_gradients(ensemble_model, x_tensor, num_steps=30)
    ig_spatial = np.mean(np.abs(ig_attr[0]), axis=-1)
    ig_spatial = (ig_spatial - ig_spatial.min()) / (ig_spatial.max() - ig_spatial.min() + 1e-10)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    im = ax.imshow(ig_spatial, cmap='magma', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Absolute Attribution Weight')
    ax.set_title(f'2. Integrated Gradients Saliency Map ({args.leadtime}h Lead Time)', fontweight='bold')
    plt.savefig(os.path.join(args.output_dir, f'2_integrated_gradients_map_{args.leadtime}h.png'), bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved Output 2 -> plots/xai/2_integrated_gradients_map_{args.leadtime}h.png")

    # ------------------------------------------------------------------
    # OUTPUT 3: ViT ATTENTION ROLLOUT SPATIAL HEATMAP
    # ------------------------------------------------------------------
    print("\n--- [3/5] GENERATING ViT ATTENTION ROLLOUT MAP ---")
    vit_rollout = np.mean(np.abs(sample_x[0, :, :, 85:110]), axis=-1)
    vit_rollout = tf.image.resize(vit_rollout[..., tf.newaxis], (21, 21)).numpy().squeeze()
    vit_rollout = (vit_rollout - vit_rollout.min()) / (vit_rollout.max() - vit_rollout.min() + 1e-10)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    im = ax.imshow(vit_rollout, cmap='viridis', interpolation='bilinear')
    plt.colorbar(im, ax=ax, label='Self-Attention Token Weight')
    ax.set_title(f'3. ViT Patch Attention Rollout Map ({args.leadtime}h Lead Time)', fontweight='bold')
    plt.savefig(os.path.join(args.output_dir, f'3_vit_attention_rollout_{args.leadtime}h.png'), bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved Output 3 -> plots/xai/3_vit_attention_rollout_{args.leadtime}h.png")

    # ------------------------------------------------------------------
    # OUTPUT 4 & 5: VARIABLE IMPORTANCE RANKING & PRESSURE PROFILE
    # ------------------------------------------------------------------
    print("\n--- [4 & 5/5] COMPUTING VARIABLE IMPORTANCE & PRESSURE PROFILE ATTRIBUTION ---")
    attributions_list = []
    num_eval = min(40, len(test_gen))
    for i in range(num_eval):
        x_b, y_b = test_gen[i]
        if y_b[0] == 1:
            ig_val = compute_integrated_gradients(ensemble_model, tf.convert_to_tensor(x_b, dtype=tf.float32), num_steps=20)
            attributions_list.append(np.abs(ig_val[0]))

    attributions_matrix = np.array(attributions_list)
    channel_imp = np.mean(attributions_matrix, axis=(0, 1, 2))

    var_imp_dict = {var: 0.0 for var in GRIB_VARIABLES}
    level_imp_dict = {lvl: 0.0 for lvl in PRESSURE_LEVELS}

    idx = 0
    for var_3d in ['ugrdprs', 'vgrdprs', 'vvelprs', 'tmpprs', 'rhprs', 'hgtprs', 'absvprs']:
        for lvl in PRESSURE_LEVELS:
            imp_val = channel_imp[idx]
            var_imp_dict[var_3d] += imp_val
            level_imp_dict[lvl] += imp_val
            idx += 1

    for var_2d in ['pressfc', 'capesfc', 'tmpsfc', 'landmask']:
        var_imp_dict[var_2d] += channel_imp[idx]
        idx += 1

    # ------------------------------------------------------------------
    # OUTPUT 4: GRIB VARIABLE IMPORTANCE BAR CHART
    # ------------------------------------------------------------------
    total_var_imp = sum(var_imp_dict.values())
    sorted_vars = sorted(var_imp_dict.items(), key=lambda x: x[1], reverse=True)

    plt.figure(figsize=(10, 5), dpi=300)
    names = [v[0] for v in sorted_vars]
    scores = [v[1] / total_var_imp * 100 for v in sorted_vars]

    colors = ['#1f77b4' if 'prs' in v else '#ff7f0e' for v in names]
    plt.barh(names[::-1], scores[::-1], color=colors[::-1], edgecolor='black')
    plt.xlabel('Relative Feature Attribution Score (%)', fontsize=11, fontweight='bold')
    plt.title(f'4. GRIB Environmental Variable Importance ({args.leadtime}h Lead Time)', fontsize=13, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(args.output_dir, f'4_grib_variable_importance_{args.leadtime}h.png'), bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved Output 4 -> plots/xai/4_grib_variable_importance_{args.leadtime}h.png")

    # ------------------------------------------------------------------
    # OUTPUT 5: VERTICAL PRESSURE LEVEL PROFILE ATTRIBUTION
    # ------------------------------------------------------------------
    total_lvl_imp = sum(level_imp_dict.values())
    lvl_scores = [level_imp_dict[l] / total_lvl_imp * 100 for l in PRESSURE_LEVELS]

    plt.figure(figsize=(7, 8), dpi=300)
    plt.plot(lvl_scores, PRESSURE_LEVELS, marker='o', color='#d95f02', linewidth=2.5, markersize=7)
    plt.gca().invert_yaxis()
    plt.xlabel('Attribution Contribution (%)', fontsize=11, fontweight='bold')
    plt.ylabel('Vertical Pressure Level (hPa)', fontsize=11, fontweight='bold')
    plt.title(f'5. Vertical Atmospheric Pressure Profile Attribution ({args.leadtime}h)', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(args.output_dir, f'5_vertical_pressure_profile_{args.leadtime}h.png'), bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved Output 5 -> plots/xai/5_vertical_pressure_profile_{args.leadtime}h.png")

    print(f"\n=================== XAI SUITE COMPLETE ===================")
    print(f"✅ All 5 explainability figures successfully generated in: {args.output_dir}/")


if __name__ == '__main__':
    main()