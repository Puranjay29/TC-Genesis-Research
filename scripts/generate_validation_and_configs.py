#!/usr/bin/env python3
"""
Generate validation predictions for one lead time and save:
- validation_predictions_log_<leadtime>h.csv
- saved_models/training_metadata.json
- saved_models/operational_thresholds.json
- saved_models/threshold_selection_report.json

This script imports the project's existing TropicalCyclogenesisGenerator and
build_advanced_ensemble so the validation preprocessing matches training.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

BASE = Path("/workspace")
SCRIPTS = BASE / "scripts"
MODELS = BASE / "saved_models"

for p in (str(BASE), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from train_ensemble import build_advanced_ensemble
from tcdl_generator import TropicalCyclogenesisGenerator


def make_generator(csv_path: Path, features_dir: Path, leadtime: int, batch_size: int):
    """Build the existing project generator by matching its actual signature."""
    signature = inspect.signature(TropicalCyclogenesisGenerator.__init__)

    values: Dict[str, Any] = {
        "csv_file": str(csv_path),
        "csv_path": str(csv_path),
        "metadata_csv": str(csv_path),
        "labels_csv": str(csv_path),
        "split_csv": str(csv_path),
        "dataframe_path": str(csv_path),
        "file_path": str(csv_path),

        "features_dir": str(features_dir),
        "feature_dir": str(features_dir),
        "data_dir": str(features_dir),
        "root_dir": str(features_dir),
        "base_dir": str(features_dir),
        "observations_dir": str(features_dir),

        "batch_size": batch_size,
        "patch_size": 21,
        "neg_to_pos_ratio": 3,
        "exclusion_radius_km": 550.0,
        "is_train": False,
        "enable_augmentation": False,
        "stats_dir": str(MODELS),
        "num_workers": 1,
        "shuffle": False,
        "augment": False,
        "augmentation": False,
        "is_training": False,
        "training": False,
        "mode": "validation",
        "split": "validation",
        "subset": "validation",
        "leadtime": leadtime,
        "lead_time": leadtime,

        "means_path": str(MODELS / "channel_means.npy"),
        "stds_path": str(MODELS / "channel_stds.npy"),
        "channel_means_path": str(MODELS / "channel_means.npy"),
        "channel_stds_path": str(MODELS / "channel_stds.npy"),
        "normalization_dir": str(MODELS),
    }

    kwargs: Dict[str, Any] = {}
    unsupported: List[str] = []

    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if name in values:
            kwargs[name] = values[name]
        elif (
            parameter.default is inspect.Parameter.empty
            and parameter.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ):
            unsupported.append(name)

    if unsupported:
        raise TypeError(
            "Could not automatically map required generator argument(s): "
            f"{unsupported}\nGenerator signature: {signature}\n"
            "Send this error and the generator class if it occurs."
        )

    print("[INFO] Generator signature:", signature)
    print("[INFO] Generator kwargs:", kwargs)
    return TropicalCyclogenesisGenerator(**kwargs)


def unpack_batch(batch: Any) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(batch, Mapping):
        x = next((batch[k] for k in ("x", "inputs", "features", "images") if k in batch), None)
        y = next((batch[k] for k in ("y", "labels", "targets", "label") if k in batch), None)
        if x is None or y is None:
            raise ValueError(f"Cannot unpack dictionary batch with keys {list(batch.keys())}")
        return np.asarray(x), np.asarray(y)

    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return np.asarray(batch[0]), np.asarray(batch[1])

    raise ValueError("Generator must return (x, y), (x, y, weights), or a dictionary.")


def run_predictions(generator, model: tf.keras.Model) -> Tuple[np.ndarray, np.ndarray]:
    all_labels: List[np.ndarray] = []
    all_probabilities: List[np.ndarray] = []

    total = len(generator)
    if total < 1:
        raise ValueError("Validation generator has zero batches.")

    for index in range(total):
        x_batch, y_batch = unpack_batch(generator[index])
        probability = np.asarray(model.predict(x_batch, verbose=0)).reshape(-1)
        label = np.asarray(y_batch).reshape(-1)

        if len(probability) != len(label):
            raise ValueError(
                f"Batch {index}: prediction count {len(probability)} "
                f"does not match label count {len(label)}."
            )

        all_labels.append(label)
        all_probabilities.append(probability)
        print(f"\r[INFO] Validation batches: {index + 1}/{total}", end="", flush=True)

    print()

    labels = np.concatenate(all_labels).astype(np.int32)
    probabilities = np.concatenate(all_probabilities).astype(np.float64)

    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(f"Labels are not binary: {np.unique(labels)}")
    if len(np.unique(labels)) != 2:
        raise ValueError("Validation split must contain both positive and negative samples.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Prediction probabilities contain NaN/Inf.")
    if probabilities.min() < 0 or probabilities.max() > 1:
        raise ValueError(
            f"Expected probabilities in [0,1], observed "
            f"[{probabilities.min()}, {probabilities.max()}]"
        )

    return labels, probabilities


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    method: str,
    target_recall: float,
) -> Tuple[float, Dict[str, Any]]:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    p = precision[:-1]
    r = recall[:-1]

    if thresholds.size == 0:
        raise ValueError("No thresholds could be calculated.")

    f1 = 2 * p * r / np.maximum(p + r, 1e-12)

    if method == "max_f1":
        idx = int(np.nanargmax(f1))
    else:
        valid = np.where(r >= target_recall)[0]
        if valid.size == 0:
            raise ValueError(f"No threshold achieved recall >= {target_recall:.3f}.")
        max_precision = np.max(p[valid])
        tied = valid[np.isclose(p[valid], max_precision)]
        idx = int(tied[-1])

    threshold = float(thresholds[idx])
    predicted = (probabilities >= threshold).astype(np.int32)

    tp = int(np.sum((labels == 1) & (predicted == 1)))
    tn = int(np.sum((labels == 0) & (predicted == 0)))
    fp = int(np.sum((labels == 0) & (predicted == 1)))
    fn = int(np.sum((labels == 1) & (predicted == 0)))

    selected_precision = tp / max(tp + fp, 1)
    selected_recall = tp / max(tp + fn, 1)
    selected_f1 = (
        2 * selected_precision * selected_recall
        / max(selected_precision + selected_recall, 1e-12)
    )

    report = {
        "selection_method": method,
        "target_recall": target_recall if method == "min_recall" else None,
        "threshold": threshold,
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "precision": float(selected_precision),
        "recall": float(selected_recall),
        "f1": float(selected_f1),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "positive_samples": int(np.sum(labels == 1)),
        "negative_samples": int(np.sum(labels == 0)),
    }
    return threshold, report


def save_prediction_log(
    validation_csv: Path,
    output_csv: Path,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    output = pd.DataFrame({
        "Sample_ID": np.arange(1, len(labels) + 1),
        "Actual_Label": labels,
        "Prob_Ensemble": probabilities,
        "Pred_Ensemble_Class_0p5": (probabilities >= 0.5).astype(np.int32),
    })

    try:
        source = pd.read_csv(validation_csv)
        if len(source) == len(output):
            for column in source.columns:
                if column not in output.columns:
                    output[column] = source[column].values
        else:
            print(
                "[WARNING] Source CSV rows do not match generated predictions; "
                "source metadata was not appended."
            )
    except Exception as exc:
        print(f"[WARNING] Could not append source metadata: {exc}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)


def save_runtime_files(output_dir: Path, leadtime: int, threshold: float, report: Dict[str, Any]):
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "dataset": "NCEP_NCAR",
        "grid_resolution_deg": 1.0,
        "patch_size": 21,
        "num_channels": 137,
        "channel_order": "variable_major_then_pressure_level",
        "pressure_variables": [
            "ugrdprs", "vgrdprs", "vvelprs", "tmpprs",
            "rhprs", "hgtprs", "absvprs"
        ],
        "pressure_levels_hpa": [
            1000, 975, 950, 925, 900, 850, 800, 750, 700, 650,
            600, 550, 500, 450, 400, 350, 300, 250, 200
        ],
        "surface_variables": ["pressfc", "capesfc", "tmpsfc", "landmask"],
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    threshold_path = output_dir / "operational_thresholds.json"
    thresholds: Dict[str, float] = {}
    if threshold_path.exists():
        loaded = json.loads(threshold_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            thresholds = {str(k): float(v) for k, v in loaded.items()}
    thresholds[str(leadtime)] = round(threshold, 8)
    threshold_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    report_path = output_dir / "threshold_selection_report.json"
    reports: Dict[str, Any] = {}
    if report_path.exists():
        loaded = json.loads(report_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            reports = loaded
    reports[str(leadtime)] = report
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Generate validation predictions and operational threshold files."
    )
    parser.add_argument("--leadtime", type=int, default=24)
    parser.add_argument(
        "--validation-csv",
        type=Path,
        default=BASE / "data/extracted_features_entire_world/tc_24h_val.csv",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=BASE / "data/extracted_features_entire_world",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--selection",
        choices=("min_recall", "max_f1"),
        default="min_recall",
    )
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--output-dir", type=Path, default=MODELS)
    parser.add_argument("--validation-output", type=Path, default=None)
    args = parser.parse_args()

    validation_csv = args.validation_csv.resolve()
    features_dir = args.features_dir.resolve()
    output_dir = args.output_dir.resolve()
    validation_output = (
        args.validation_output.resolve()
        if args.validation_output
        else features_dir / f"validation_predictions_log_{args.leadtime}h.csv"
    )

    if not validation_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {validation_csv}")

    cnn = MODELS / f"best_cnn_lstm_{args.leadtime}h.keras"
    vit = MODELS / f"best_vit_gru_{args.leadtime}h.keras"
    ensemble_weights = MODELS / f"best_ensemble_{args.leadtime}h.keras"

    missing = [p for p in (cnn, vit, ensemble_weights) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(str(p) for p in missing)
        )

    print(f"[INFO] Building {args.leadtime} h ensemble...")
    ensemble_model, _, _ = build_advanced_ensemble(
        str(cnn), str(vit), input_shape=(21, 21, 137)
    )
    ensemble_model.load_weights(str(ensemble_weights))

    generator = make_generator(
        validation_csv, features_dir, args.leadtime, args.batch_size
    )
    labels, probabilities = run_predictions(generator, ensemble_model)

    save_prediction_log(
        validation_csv, validation_output, labels, probabilities
    )
    print(f"[SAVED] {validation_output}")

    threshold, report = select_threshold(
        labels, probabilities, args.selection, args.target_recall
    )
    save_runtime_files(output_dir, args.leadtime, threshold, report)

    print()
    print(f"=== {args.leadtime}-HOUR VALIDATION RESULTS ===")
    print(f"Threshold       : {threshold:.8f}")
    print(f"PR-AUC          : {report['pr_auc']:.6f}")
    print(f"ROC-AUC         : {report['roc_auc']:.6f}")
    print(f"Precision       : {report['precision']:.6f}")
    print(f"Recall          : {report['recall']:.6f}")
    print(f"F1              : {report['f1']:.6f}")
    print(f"TP/TN/FP/FN     : {report['true_positive']}/"
          f"{report['true_negative']}/"
          f"{report['false_positive']}/"
          f"{report['false_negative']}")
    print()
    print(f"[SAVED] {output_dir / 'training_metadata.json'}")
    print(f"[SAVED] {output_dir / 'operational_thresholds.json'}")
    print(f"[SAVED] {output_dir / 'threshold_selection_report.json'}")


if __name__ == "__main__":
    main()