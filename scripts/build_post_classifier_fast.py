#!/usr/bin/env python3
"""
build_post_classifier_fast.py

Faster all-in-one post-classifier builder.

What is faster:
- Reuses cached FNL files.
- Uses the parallel verification script.
- Runs month-by-month and stops as soon as enough labelled detections exist.
- Starts with the more active July-November period.
- Reuses an existing enriched CSV.
- Calculates XAI in batches per FNL file instead of one hotspot at a time.
- Trains immediately once the target sample count and class-balance checks are met.

The learned JSON is installed only when cross-validated precision and CSI are
not worse than the six-condition fallback. Otherwise main.py auto mode keeps
using the hardcoded rules.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


XAI_VARS = [
    "ugrdprs", "vgrdprs", "vvelprs", "tmpprs", "rhprs", "hgtprs",
    "absvprs", "pressfc", "capesfc", "tmpsfc", "landmask",
]

FEATURES = [
    "ensemble_score",
    "cnn_score",
    "vit_score",
    "land_fraction",
    "pressure_anomaly_pa",
    "midlevel_rh_percent",
    "shear_200_850_ms",
    "vorticity_850_s1",
    "cyclonic_vorticity_850_s1",
    "surface_temperature_k",
    "matched_distance_km",
    "matched_age_hours",
] + [f"xai_{name}" for name in XAI_VARS]


def log(message: str) -> None:
    print(f"[FAST-BUILDER] {message}", flush=True)


def month_pairs(start_year: int, start_month: int, months: int) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    y, m = start_year, start_month
    for _ in range(months):
        start = date(y, m, 1)
        if m == 12:
            end = date(y + 1, 1, 1)
            y, m = y + 1, 1
        else:
            end = date(y, m + 1, 1)
            m += 1
        result.append((start.isoformat(), end.isoformat()))
    return result


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("fnl_verifier_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prediction_csvs(*roots: Path) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*_predictions.csv"):
            resolved = path.resolve()
            if resolved not in seen:
                found.append(path)
                seen.add(resolved)
    return sorted(found)


def normalize_frame(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    frame = frame.copy()
    frame["source_csv"] = str(source)

    if "verification_status" not in frame.columns:
        return frame.iloc[0:0].copy()

    status = frame["verification_status"].astype(str).str.upper().str.strip()
    frame = frame[status.isin(["HIT", "FALSE_ALARM"])].copy()
    frame["is_genesis_hit"] = (
        frame["verification_status"].astype(str).str.upper().str.strip() == "HIT"
    ).astype(int)

    aliases = {
        "confidence": "ensemble_score",
        "raw_model_score": "ensemble_score",
        "cnn_confidence": "cnn_score",
        "vit_confidence": "vit_score",
        "sst_k": "surface_temperature_k",
        "matched_system_distance_km": "matched_distance_km",
        "matched_system_age_hours": "matched_age_hours",
    }
    for source_col, target_col in aliases.items():
        if target_col not in frame.columns and source_col in frame.columns:
            frame[target_col] = frame[source_col]

    return frame


def combine_csvs(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = normalize_frame(pd.read_csv(path), path)
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            log(f"Skipping unreadable CSV {path}: {exc}")

    if not frames:
        return pd.DataFrame()

    frame = pd.concat(frames, ignore_index=True, sort=False)

    dedupe = [
        c for c in (
            "analysis_time_utc", "file", "latitude", "longitude",
            "verification_status", "confidence",
        )
        if c in frame.columns
    ]
    if dedupe:
        frame = frame.drop_duplicates(subset=dedupe, keep="last")

    return frame.reset_index(drop=True)


def class_counts(frame: pd.DataFrame) -> tuple[int, int, int]:
    if frame.empty or "is_genesis_hit" not in frame.columns:
        return 0, 0, 0
    hits = int(frame["is_genesis_hit"].sum())
    false_alarms = int((frame["is_genesis_hit"] == 0).sum())
    return len(frame), hits, false_alarms


def enough(frame: pd.DataFrame, args: argparse.Namespace) -> bool:
    total, hits, false_alarms = class_counts(frame)
    return (
        total >= args.target_samples
        and hits >= args.minimum_hits
        and false_alarms >= args.minimum_false_alarms
    )


def run_month(args: argparse.Namespace, start_date: str, end_date: str) -> None:
    tag = start_date[:7].replace("-", "")
    out_dir = args.new_output_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(args.verifier_script),
        "--leadtime", "24",
        "--ibtracs-csv", str(args.ibtracs_csv),
        "--start-date", start_date,
        "--end-date", end_date,
        "--genesis-definition", "first-track",
        "--lead-tolerance-hours", "0",
        "--alert-threshold", "0.75",
        "--min-expert-confidence", "0.45",
        "--minimum-midlevel-rh-percent", "50",
        "--minimum-vorticity-850-s1", "7.5e-6",
        "--maximum-pressure-anomaly-pa", "0",
        "--maximum-shear-200-850-ms", "25",
        "--maximum-lowlevel-divergence-s1=1.0e-4",
        "--min-cluster-support", "2",
        "--cluster-radius-km", "700",
        "--max-detections", "2",
        "--ibtracs-distance-tolerance-km", "800",
        "--batch-size", str(args.batch_size),
        "--download-workers", str(args.download_workers),
        "--fnl-cache-dir", str(args.fnl_cache_dir),
        "--output-dir", str(out_dir),
    ]
    if args.insecure_ssl:
        command.append("--insecure-ssl")

    log(f"Running {start_date} to {end_date}")
    result = subprocess.run(command, cwd=str(args.project_dir))
    if result.returncode != 0:
        raise RuntimeError(
            f"Verification failed for {start_date} to {end_date} "
            f"with exit code {result.returncode}"
        )


def find_grib(cache_dir: Path, file_value: Any, analysis_time: Any) -> Path | None:
    candidates: list[str] = []

    if pd.notna(file_value):
        value = str(file_value).strip()
        if value:
            candidates.extend([value, value + ".grib2", value + ".grb2"])

    if pd.notna(analysis_time):
        try:
            stamp = pd.Timestamp(analysis_time)
            candidates.extend([
                f"fnl_{stamp:%Y%m%d_%H}_00.grib2",
                f"fnl_{stamp:%Y%m%d_%H}_00.grb2",
            ])
        except Exception:
            pass

    for candidate in candidates:
        direct = cache_dir / candidate
        if direct.exists():
            return direct
        matches = list(cache_dir.rglob(candidate))
        if matches:
            return matches[0]
    return None


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.nanargmin(np.abs(values.astype(float) - float(target))))


def wrapped_lon_index(values: np.ndarray, target: float) -> int:
    distances = np.abs((values.astype(float) - target + 180.0) % 360.0 - 180.0)
    return int(np.nanargmin(distances))


def patch_mean(field: np.ndarray, lat_idx: int, lon_idx: int, half: int = 10) -> float:
    lat0 = max(0, lat_idx - half)
    lat1 = min(field.shape[0], lat_idx + half + 1)
    lon_ids = np.mod(np.arange(lon_idx - half, lon_idx + half + 1), field.shape[1])
    return float(np.nanmean(field[lat0:lat1][:, lon_ids]))


def batch_xai(model: tf.keras.Model, patches: np.ndarray, batch_size: int) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []

    for start in range(0, len(patches), batch_size):
        chunk = tf.convert_to_tensor(
            patches[start:start + batch_size],
            dtype=tf.float32,
        )
        with tf.GradientTape() as tape:
            tape.watch(chunk)
            pred = model(chunk, training=False)
            pred = tf.reshape(pred, [-1])
            objective = tf.reduce_sum(pred)

        grads = tape.gradient(objective, chunk)
        if grads is None:
            raise RuntimeError("No input gradients returned by TensorFlow.")

        importance = np.mean(np.abs(grads.numpy()), axis=(1, 2))

        for vector in importance:
            values: dict[str, float] = {}
            index = 0
            for variable in XAI_VARS[:7]:
                values[f"xai_{variable}"] = float(np.mean(vector[index:index + 19]))
                index += 19
            for variable in XAI_VARS[7:]:
                values[f"xai_{variable}"] = float(vector[index])
                index += 1

            total = sum(values.values())
            if total > 0:
                values = {k: v / total for k, v in values.items()}
            results.append(values)

    return results


def enrich_missing_rows(
    raw: pd.DataFrame,
    existing_enriched: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    key_cols = [
        c for c in (
            "analysis_time_utc", "file", "latitude", "longitude",
            "verification_status",
        )
        if c in raw.columns
    ]

    if not existing_enriched.empty and key_cols:
        old_keys = existing_enriched[key_cols].astype(str).agg("|".join, axis=1)
        old_map = dict(zip(old_keys, existing_enriched.to_dict("records")))
    else:
        old_map = {}

    raw_keys = (
        raw[key_cols].astype(str).agg("|".join, axis=1)
        if key_cols
        else pd.Series([""] * len(raw))
    )

    reused: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for key, record in zip(raw_keys, raw.to_dict("records")):
        cached = old_map.get(key)
        if cached and any(
            pd.notna(cached.get(f"xai_{v}"))
            for v in XAI_VARS
        ):
            reused.append(cached)
        else:
            pending.append(record)

    log(f"Reusing {len(reused)} enriched rows; computing {len(pending)} new rows.")

    if not pending:
        return pd.DataFrame(reused)

    verifier = load_module(args.verifier_script)
    means, stds = verifier.load_normalization()
    ensemble, _, _, _ = verifier.load_models(24)

    pending_frame = pd.DataFrame(pending)
    output: list[dict[str, Any]] = list(reused)

    group_columns = [c for c in ("file", "analysis_time_utc") if c in pending_frame.columns]
    grouped = pending_frame.groupby(group_columns, dropna=False, sort=False)

    for group_index, (group_key, group) in enumerate(grouped, 1):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        file_value = group.iloc[0].get("file")
        analysis_time = group.iloc[0].get("analysis_time_utc")

        grib = find_grib(args.fnl_cache_dir, file_value, analysis_time)
        if grib is None:
            log(f"Missing cached FNL for {group_key}; rows retained without XAI.")
            output.extend(group.to_dict("records"))
            continue

        log(f"XAI {group_index}/{len(grouped)}: {grib.name}, {len(group)} detections")

        ds = verifier.preprocess_grib(grib)
        grid = verifier.construct_grid(ds, means, stds)
        padded = np.pad(
            grid,
            ((0, 0), (verifier.HALF_PATCH, verifier.HALF_PATCH), (0, 0)),
            mode="wrap",
        )

        lat_values = np.asarray(ds["lat"].values, dtype=float)
        lon_values = np.asarray(ds["lon"].values, dtype=float)

        u200 = np.asarray(ds["ugrdprs"].sel(lev=200).values, dtype=float)
        v200 = np.asarray(ds["vgrdprs"].sel(lev=200).values, dtype=float)
        u850 = np.asarray(ds["ugrdprs"].sel(lev=850).values, dtype=float)
        v850 = np.asarray(ds["vgrdprs"].sel(lev=850).values, dtype=float)
        sst = np.asarray(ds["tmpsfc"].values, dtype=float)

        records = group.to_dict("records")
        indices: list[list[int]] = []
        for rec in records:
            lat = float(rec["latitude"])
            lon = float(rec["longitude"]) % 360.0
            indices.append([
                nearest_index(lat_values, lat),
                wrapped_lon_index(lon_values, lon),
            ])

        idx = np.asarray(indices, dtype=np.int64)
        patches = verifier.extract_patch_batch(padded, idx)
        xai_values = batch_xai(ensemble, patches, args.xai_batch_size)

        for rec, (lat_idx, lon_idx), xai in zip(records, indices, xai_values):
            du = patch_mean(u200 - u850, lat_idx, lon_idx)
            dv = patch_mean(v200 - v850, lat_idx, lon_idx)
            rec["shear_200_850_ms"] = float(math.hypot(du, dv))
            rec["surface_temperature_k"] = patch_mean(sst, lat_idx, lon_idx)

            latitude = float(rec["latitude"])
            raw_vort = pd.to_numeric(
                pd.Series([rec.get("vorticity_850_s1")]),
                errors="coerce",
            ).iloc[0]
            if pd.notna(raw_vort):
                rec["cyclonic_vorticity_850_s1"] = (
                    float(raw_vort) if latitude >= 0 else -float(raw_vort)
                )

            rec.update(xai)
            rec["xai_available"] = True
            output.append(rec)

        del ds, grid, padded, patches

    return pd.DataFrame(output)


def csi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp / max(tp + fp + fn, 1)


def fallback_prediction(frame: pd.DataFrame) -> np.ndarray:
    values = lambda name, fill: pd.to_numeric(
        frame[name], errors="coerce"
    ).fillna(fill).to_numpy()

    checks = np.column_stack([
        values("land_fraction", 1.0) <= 0.15,
        values("midlevel_rh_percent", 0.0) >= 55.0,
        values("shear_200_850_ms", 999.0) <= 20.0,
        values("cyclonic_vorticity_850_s1", 0.0) >= 7.5e-6,
        values("pressure_anomaly_pa", 9999.0) <= 0.0,
        values("surface_temperature_k", 0.0) >= 299.0,
    ])
    return (checks.sum(axis=1) >= 5).astype(int)


def train(frame: pd.DataFrame, args: argparse.Namespace) -> bool:
    total, hits, false_alarms = class_counts(frame)
    log(f"Training rows={total}, hits={hits}, false alarms={false_alarms}")

    if not enough(frame, args):
        log("Not enough balanced labelled rows; learned JSON will not be installed.")
        return False

    y = frame["is_genesis_hit"].astype(int).to_numpy()
    selected = [
        feature for feature in FEATURES
        if feature in frame.columns and frame[feature].notna().any()
    ]

    required = {
        "ensemble_score", "cnn_score", "vit_score",
        "land_fraction", "pressure_anomaly_pa",
        "midlevel_rh_percent", "shear_200_850_ms",
        "cyclonic_vorticity_850_s1", "surface_temperature_k",
    }
    if required - set(selected):
        log(f"Missing features: {sorted(required - set(selected))}")
        return False

    if len([f for f in selected if f.startswith("xai_")]) < 7:
        log("Fewer than seven XAI feature groups are available.")
        return False

    X = frame[selected].apply(pd.to_numeric, errors="coerce").to_numpy()
    folds = min(args.folds, int(np.bincount(y).min()))
    if folds < 2:
        log("Insufficient examples in the smaller class.")
        return False

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=0.35,
            random_state=42,
        )),
    ])

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

    best = None
    for threshold in np.linspace(0.20, 0.80, 121):
        pred = (prob >= threshold).astype(int)
        precision = float(precision_score(y, pred, zero_division=0))
        recall = float(recall_score(y, pred, zero_division=0))
        score = float(csi(y, pred))
        candidate = (score + 0.15 * precision, threshold, precision, recall, score)
        if best is None or candidate > best:
            best = candidate

    _, threshold, learned_precision, learned_recall, learned_csi = best

    rules = fallback_prediction(frame)
    rules_precision = float(precision_score(y, rules, zero_division=0))
    rules_recall = float(recall_score(y, rules, zero_division=0))
    rules_csi = float(csi(y, rules))

    log(
        f"Learned CV: P={learned_precision:.3f}, R={learned_recall:.3f}, "
        f"CSI={learned_csi:.3f}, threshold={threshold:.3f}"
    )
    log(
        f"Fallback: P={rules_precision:.3f}, R={rules_recall:.3f}, "
        f"CSI={rules_csi:.3f}"
    )

    accepted = (
        learned_precision >= args.minimum_validation_precision
        and learned_precision >= rules_precision
        and learned_csi >= rules_csi
    )
    if not accepted:
        log("Learned model rejected; keeping six-condition fallback.")
        return False

    pipe.fit(X, y)
    imputer = pipe.named_steps["imputer"]
    scaler = pipe.named_steps["scaler"]
    model = pipe.named_steps["model"]

    payload = {
        "model_type": "logistic_regression",
        "uses_xai": True,
        "feature_names": selected,
        "coefficients": dict(zip(selected, map(float, model.coef_[0]))),
        "intercept": float(model.intercept_[0]),
        "means": dict(zip(selected, map(float, scaler.mean_))),
        "scales": dict(zip(
            selected,
            [float(v if v > 1e-12 else 1.0) for v in scaler.scale_],
        )),
        "imputation_medians": dict(zip(
            selected,
            map(float, imputer.statistics_),
        )),
        "positive_threshold": float(threshold),
        "high_priority_threshold": float(min(0.90, threshold + 0.15)),
        "negative_threshold": float(max(0.10, threshold - 0.20)),
        "maximum_absolute_z": 5.0,
        "minimum_validation_samples": int(args.target_samples),
        "minimum_validation_precision": float(args.minimum_validation_precision),
        "training_samples": int(total),
        "validation_metrics": {
            "hits": hits,
            "false_alarms": false_alarms,
            "precision": learned_precision,
            "recall": learned_recall,
            "csi": learned_csi,
            "rules_precision": rules_precision,
            "rules_recall": rules_recall,
            "rules_csi": rules_csi,
            "folds": int(folds),
        },
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.json_output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(args.json_output)
    log(f"Installed learned post-classifier: {args.json_output}")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--project-dir", type=Path, default=Path("/workspace"))
    parser.add_argument(
        "--verifier-script",
        type=Path,
        default=Path(
            "/workspace/scripts/"
            "run_fnl_2026_cyclogenesis_verification_parallel.py"
        ),
    )
    parser.add_argument(
        "--ibtracs-csv",
        type=Path,
        default=Path(
            "/workspace/data/ibtracs/ibtracs.ALL.list.v04r01.csv"
        ),
    )
    parser.add_argument(
        "--existing-results-dir",
        type=Path,
        default=Path("/workspace/data/fnl_realworld_verification"),
    )
    parser.add_argument(
        "--new-output-root",
        type=Path,
        default=Path("/workspace/data/fnl_post_classifier_fast"),
    )
    parser.add_argument(
        "--fnl-cache-dir",
        type=Path,
        default=Path("/workspace/data/fnl_cache"),
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=Path(
            "/workspace/data/post_classifier_training/"
            "genesis_post_classifier_training.csv"
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "/workspace/saved_models/genesis_post_classifier.json"
        ),
    )

    parser.add_argument("--start-year", type=int, default=2025)
    parser.add_argument("--start-month", type=int, default=7)
    parser.add_argument("--months", type=int, default=5)
    parser.add_argument("--target-samples", type=int, default=100)
    parser.add_argument("--minimum-hits", type=int, default=20)
    parser.add_argument("--minimum-false-alarms", type=int, default=40)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--minimum-validation-precision", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--xai-batch-size", type=int, default=16)
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--insecure-ssl", action="store_true")
    parser.add_argument("--skip-verification", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    for name in (
        "project_dir", "verifier_script", "ibtracs_csv",
        "existing_results_dir", "new_output_root",
        "fnl_cache_dir", "combined_csv", "json_output",
    ):
        setattr(args, name, getattr(args, name).resolve())

    if not args.verifier_script.exists():
        raise SystemExit(f"Verifier not found: {args.verifier_script}")
    if not args.ibtracs_csv.exists():
        raise SystemExit(f"IBTrACS not found: {args.ibtracs_csv}")

    existing_enriched = (
        pd.read_csv(args.combined_csv)
        if args.combined_csv.exists()
        else pd.DataFrame()
    )

    paths = prediction_csvs(args.existing_results_dir, args.new_output_root)
    combined = combine_csvs(paths)
    total, hits, false_alarms = class_counts(combined)
    log(f"Existing labelled rows={total}, hits={hits}, false alarms={false_alarms}")

    if not args.skip_verification and not enough(combined, args):
        for start_date, end_date in month_pairs(
            args.start_year,
            args.start_month,
            args.months,
        ):
            month_dir = args.new_output_root / start_date[:7].replace("-", "")
            if prediction_csvs(month_dir):
                log(f"Skipping completed month {start_date[:7]}")
            else:
                run_month(args, start_date, end_date)

            paths = prediction_csvs(args.existing_results_dir, args.new_output_root)
            combined = combine_csvs(paths)
            total, hits, false_alarms = class_counts(combined)
            log(
                f"After {start_date[:7]}: rows={total}, "
                f"hits={hits}, false alarms={false_alarms}"
            )
            if enough(combined, args):
                log("Target reached; stopping additional historical verification.")
                break

    if combined.empty:
        raise SystemExit("No labelled prediction rows available.")

    enriched = enrich_missing_rows(combined, existing_enriched, args)
    args.combined_csv.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.combined_csv, index=False)
    log(f"Saved training CSV: {args.combined_csv}")

    accepted = train(enriched, args)

    print()
    print("=" * 72)
    if accepted:
        print("POST-CLASSIFIER READY")
        print(f"JSON: {args.json_output}")
        print("Run main.py with --classification-mode auto")
    else:
        print("LEARNED MODEL NOT INSTALLED")
        print("main.py auto mode will continue with the six-condition fallback.")
    print("=" * 72)


if __name__ == "__main__":
    main()
