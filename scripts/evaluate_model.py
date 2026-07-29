#!/usr/bin/env python3

"""
Out-of-Sample Evaluation Script for Operational Tropical Cyclogenesis Detection Models.

Evaluates trained model checkpoints (CNN-LSTM, ViT-BiGRU, Ensemble) on unseen test splits
and generates classification reports, PR-AUC, ROC-AUC, and confusion matrices.
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    auc
)
from tcdl_generator import TropicalCyclogenesisGenerator
from train_vit_gru import PatchExtractorAndEmbedder, SqueezeAndExcitationBlock
from train_ensemble import LayerScale


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Operational TC Genesis Model on Test Split.")
    parser.add_argument('--leadtime', type=int, default=24, choices=[24, 48, 72])
    parser.add_argument('--model-type', type=str, default='ensemble', choices=['cnn_lstm', 'vit_gru', 'ensemble'])
    parser.add_argument('--batch-size', type=int, default=36)
    parser.add_argument('--stats-dir', type=str, default='saved_models')
    return parser.parse_args()


def main():
    args = parse_args()
    test_csv = f"data/extracted_features_entire_world/tc_{args.leadtime}h_test.csv"
    model_path = f"saved_models/best_{args.model_type}_{args.leadtime}h.keras"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test split CSV not found: {test_csv}")

    print(f"\n=================== EVALUATING MODEL ({args.model_type.upper()} | {args.leadtime}h LEAD TIME) ===================")
    print(f"  • Model Checkpoint : {model_path}")
    print(f"  • Test Dataset CSV : {test_csv}")

    # Load holdout test generator (deterministic mode, no augmentation)
    test_gen = TropicalCyclogenesisGenerator(
        test_csv,
        batch_size=args.batch_size,
        patch_size=21,
        is_train=False,
        enable_augmentation=False,
        stats_dir=args.stats_dir
    )

    print(f"\n--> Loading Trained Model Keras Weights...")
    custom_objects = {
        'SqueezeAndExcitationBlock': SqueezeAndExcitationBlock,
        'PatchExtractorAndEmbedder': PatchExtractorAndEmbedder,
        'LayerScale': LayerScale
    }
    
    # safe_mode=False allows loading custom Lambda functions inside the Ensemble
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False, safe_mode=False)

    print("--> Running Inference over Holdout Test Set...")
    y_true = test_gen.samples_y
    y_pred_probs = model.predict(test_gen, verbose=1).flatten()[:len(y_true)]

    # Compute binary metrics at 0.5 decision threshold
    y_pred_binary = (y_pred_probs >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()

    precision_vec, recall_vec, _ = precision_recall_curve(y_true, y_pred_probs)
    pr_auc_val = auc(recall_vec, precision_vec)
    roc_auc_val = roc_auc_score(y_true, y_pred_probs)

    print("\n=================== HOLDOUT TEST PERFORMANCE RESULTS ===================")
    print(f"  • ROC-AUC Score : {roc_auc_val:.5f}")
    print(f"  • PR-AUC Score  : {pr_auc_val:.5f}")
    print("\n--> Confusion Matrix:")
    print(f"    True Negatives  (TN) : {tn}")
    print(f"    False Positives (FP) : {fp}")
    print(f"    False Negatives (FN) : {fn}")
    print(f"    True Positives  (TP) : {tp}")

    print("\n--> Detailed Classification Report:")
    print(classification_report(y_true, y_pred_binary, target_names=['No-Genesis', 'Genesis'], digits=4))

    # Save evaluation summary to JSON
    eval_results = {
        "leadtime": args.leadtime,
        "model_type": args.model_type,
        "test_roc_auc": float(roc_auc_val),
        "test_pr_auc": float(pr_auc_val),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "precision": float(tp / (tp + fp + 1e-6)),
        "recall": float(tp / (tp + fn + 1e-6)),
        "f1_score": float(2 * tp / (2 * tp + fp + fn + 1e-6))
    }

    out_json = f"saved_models/test_results_{args.model_type}_{args.leadtime}h.json"
    with open(out_json, 'w') as f:
        json.dump(eval_results, f, indent=4)
    print(f"\nSaved evaluation metrics to: {out_json}")


if __name__ == '__main__':
    main()