#!/usr/bin/env python3

"""
Direct Weight-Loading Evaluation Script for the Gated Ensemble.
Bypasses Keras Lambda deserialization issues by rebuilding the graph and loading weights directly.
"""

import argparse
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    auc
)
from tcdl_generator import TropicalCyclogenesisGenerator
from train_ensemble import build_advanced_ensemble


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Operational Gated Ensemble on Test Split.")
    parser.add_argument('--leadtime', type=int, default=24, choices=[24, 48, 72])
    parser.add_argument('--batch-size', type=int, default=36)
    parser.add_argument('--stats-dir', type=str, default='saved_models')
    return parser.parse_args()


def main():
    args = parse_args()
    test_csv = f"data/extracted_features_entire_world/tc_{args.leadtime}h_test.csv"
    model_path = f"saved_models/best_ensemble_{args.leadtime}h.keras"
    cnn_lstm_path = f'saved_models/best_cnn_lstm_{args.leadtime}h.keras'
    vit_gru_path  = f'saved_models/best_vit_gru_{args.leadtime}h.keras'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Ensemble model file not found: {model_path}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test split CSV not found: {test_csv}")

    print(f"\n=================== EVALUATING ENSEMBLE ({args.leadtime}h LEAD TIME) ===================")
    print(f"  • Ensemble Checkpoint : {model_path}")
    print(f"  • Test Dataset CSV     : {test_csv}")

    # Load holdout test generator
    test_gen = TropicalCyclogenesisGenerator(
        test_csv,
        batch_size=args.batch_size,
        patch_size=21,
        is_train=False,
        enable_augmentation=False,
        stats_dir=args.stats_dir
    )

    print(f"\n--> Rebuilding Ensemble Graph & Loading Pre-trained Weights directly...")
    # Reconstruct the architecture graph
    model, _, _ = build_advanced_ensemble(cnn_lstm_path, vit_gru_path, input_shape=(21, 21, 137))
    
    # Load exact weights from the saved checkpoint
    model.load_weights(model_path)
    print("--> Checkpoint weights loaded successfully!")

    print("\n--> Running Inference over Holdout Test Set...")
    y_true = test_gen.samples_y
    y_pred_probs = model.predict(test_gen, verbose=1).flatten()[:len(y_true)]

    # Compute binary metrics at 0.5 threshold
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
        "model_type": "ensemble",
        "test_roc_auc": float(roc_auc_val),
        "test_pr_auc": float(pr_auc_val),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "precision": float(tp / (tp + fp + 1e-6)),
        "recall": float(tp / (tp + fn + 1e-6)),
        "f1_score": float(2 * tp / (2 * tp + fp + fn + 1e-6))
    }

    out_json = f"saved_models/test_results_ensemble_{args.leadtime}h.json"
    with open(out_json, 'w') as f:
        json.dump(eval_results, f, indent=4)
    print(f"\nSaved evaluation metrics to: {out_json}")


if __name__ == '__main__':
    main()