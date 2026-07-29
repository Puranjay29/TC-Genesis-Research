#!/usr/bin/env python3
"""
Fit lead-time-specific Platt probability calibration from held-out validation
predictions.

This does not change model discrimination. It maps raw neural scores onto
empirical validation frequencies so that a raw score such as 0.99 is not
automatically presented as a 99% real-world formation probability.

Example:
python3 /workspace/scripts/fit_probability_calibration.py \
  --prediction 24:/workspace/data/extracted_features_entire_world/validation_predictions_log_24h.csv \
  --output /workspace/saved_models/probability_calibration.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


LABEL_CANDIDATES = (
    "Actual_Label", "actual_label", "label", "Label", "y_true", "target",
)
PROBABILITY_CANDIDATES = (
    "Prob_Ensemble", "prob_ensemble", "ensemble_probability",
    "probability", "confidence", "y_prob",
)


def parse_horizon_path(value: str) -> Tuple[int, Path]:
    try:
        horizon_text, path_text = value.split(":", 1)
        return int(horizon_text), Path(path_text)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "Expected HORIZON:PATH, for example 24:/workspace/...csv"
        ) from exc


def detect_column(frame: pd.DataFrame, candidates, explicit):
    if explicit:
        if explicit not in frame.columns:
            raise ValueError(f"Column {explicit!r} not found")
        return explicit

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    raise ValueError(
        f"Could not detect column. Available columns: {list(frame.columns)}"
    )


def logit(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(probabilities, 1.0e-6, 1.0 - 1.0e-6)
    return np.log(probabilities / (1.0 - probabilities))


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.digitize(probabilities, edges[1:-1], right=True)
    total = len(labels)
    ece = 0.0

    for bin_index in range(bins):
        mask = indices == bin_index
        count = int(mask.sum())
        if count == 0:
            continue
        observed = float(labels[mask].mean())
        predicted = float(probabilities[mask].mean())
        ece += (count / total) * abs(observed - predicted)

    return float(ece)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction",
        action="append",
        type=parse_horizon_path,
        required=True,
        help="Repeatable HORIZON:CSV validation prediction input.",
    )
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--probability-column", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/workspace/saved_models/probability_calibration.json"
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    document: Dict[str, object] = {
        "format_version": 1,
        "description": (
            "Platt calibration fitted on held-out validation predictions. "
            "Operational domain shift may reduce reliability."
        ),
        "horizons": {},
    }

    for horizon, path in args.prediction:
        frame = pd.read_csv(path)
        label_column = detect_column(
            frame, LABEL_CANDIDATES, args.label_column
        )
        probability_column = detect_column(
            frame, PROBABILITY_CANDIDATES, args.probability_column
        )

        labels = pd.to_numeric(
            frame[label_column], errors="coerce"
        ).to_numpy()
        probabilities = pd.to_numeric(
            frame[probability_column], errors="coerce"
        ).to_numpy()

        valid = (
            np.isfinite(labels)
            & np.isfinite(probabilities)
            & np.isin(labels, [0, 1])
        )
        labels = labels[valid].astype(np.int32)
        probabilities = np.clip(
            probabilities[valid].astype(np.float64),
            1.0e-6,
            1.0 - 1.0e-6,
        )

        if len(np.unique(labels)) != 2:
            raise ValueError(
                f"{horizon} h data must contain both classes"
            )

        features = logit(probabilities).reshape(-1, 1)
        minimum_class_count = int(
            min(np.bincount(labels, minlength=2))
        )
        folds = min(args.folds, minimum_class_count)
        if folds < 2:
            raise ValueError("Not enough samples for cross-validation")

        model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
        )
        cv = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=42,
        )

        oof_calibrated = cross_val_predict(
            model,
            features,
            labels,
            cv=cv,
            method="predict_proba",
        )[:, 1]

        model.fit(features, labels)
        final_calibrated = model.predict_proba(features)[:, 1]

        raw_metrics = {
            "brier": float(brier_score_loss(labels, probabilities)),
            "log_loss": float(log_loss(labels, probabilities)),
            "ece_10bin": expected_calibration_error(
                labels, probabilities
            ),
            "roc_auc": float(roc_auc_score(labels, probabilities)),
        }
        calibrated_metrics_oof = {
            "brier": float(
                brier_score_loss(labels, oof_calibrated)
            ),
            "log_loss": float(log_loss(labels, oof_calibrated)),
            "ece_10bin": expected_calibration_error(
                labels, oof_calibrated
            ),
            "roc_auc": float(
                roc_auc_score(labels, oof_calibrated)
            ),
        }

        document["horizons"][str(horizon)] = {
            "method": "platt_logit",
            "coefficient": float(model.coef_[0, 0]),
            "intercept": float(model.intercept_[0]),
            "sample_count": int(len(labels)),
            "positive_count": int(labels.sum()),
            "negative_count": int((labels == 0).sum()),
            "label_column": label_column,
            "probability_column": probability_column,
            "source_file": str(path),
            "cross_validation_folds": folds,
            "raw_metrics": raw_metrics,
            "calibrated_metrics_out_of_fold": (
                calibrated_metrics_oof
            ),
            "warning": (
                "Calibration describes the validation domain. "
                "GDAS/GFS operational domain shift remains."
            ),
        }

        print(f"\n=== {horizon} h calibration ===")
        print(
            f"coefficient={model.coef_[0, 0]:.8f} "
            f"intercept={model.intercept_[0]:.8f}"
        )
        print("Raw metrics:", raw_metrics)
        print(
            "OOF calibrated metrics:",
            calibrated_metrics_oof,
        )
        print(
            "Example mapping: "
            + ", ".join(
                f"{value:.2f}->{model.predict_proba([[float(logit(np.array([value]))[0])]])[0,1]:.3f}"
                for value in (0.60, 0.80, 0.90, 0.99)
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()