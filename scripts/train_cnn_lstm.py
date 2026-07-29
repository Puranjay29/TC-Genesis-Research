#!/usr/bin/env python3

"""
Production CNN-LSTM Training Pipeline for Tropical Cyclogenesis Prediction.

Architecture & Scientific Integrity:
  - Dual-Axis Spatial Sequence LSTM (Parallel Row-wise & Column-wise feature extraction).
  - Mixed Precision FP16 acceleration for modern NVIDIA GPUs.
  - Binary Focal Cross-Entropy Loss (gamma=2.0, alpha=0.25) to prioritize high recall.
  - Monitored via PR-AUC (Precision-Recall AUC) on Validation split.
  - Gradient clipping (clipnorm=1.0) and full random reproducibility (SEED=42).

Execution & Infrastructure Optimizations:
  - Multi-worker threading support (--workers 16) for 24-core CPU parallel I/O.
  - Automatic pre-run file and precomputed normalization stats verification.
  - Comprehensive metadata serialization (JSON) and training log exports (CSV).
  - TensorBoard integration (logs/cnn_lstm_{leadtime}h).
"""

import argparse
from datetime import datetime
import json
import os
import pickle
import random
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers, mixed_precision, models, regularizers
from tcdl_generator import TropicalCyclogenesisGenerator

# 1. Enforce Exact Reproducibility Seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 2. Enable GPU Mixed Precision Acceleration
try:
    mixed_precision.set_global_policy("mixed_float16")
    print("--> Mixed Precision policy set to 'mixed_float16' for GPU acceleration.")
except Exception as e:
    print(f"--> Could not set Mixed Precision policy: {e}")

# Prevent layout optimizer crashes on custom tensor shapes
tf.config.optimizer.set_experimental_options({"layout_optimizer": False})


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Production Dual-Axis CNN-LSTM Model.")
    parser.add_argument(
        '--leadtime',
        dest='leadtime',
        type=int,
        default=24,
        choices=[24, 48, 72],
        help='Forecast lead time in hours (24, 48, or 72). Default is 24.')
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        type=int,
        default=36,
        help='Batch size for training. Default is 36.')
    parser.add_argument(
        '--epochs',
        dest='epochs',
        type=int,
        default=100,
        help='Maximum training epochs. Default is 100.')
    parser.add_argument(
        '--workers',
        dest='num_workers',
        type=int,
        default=16,
        help='Number of parallel CPU worker threads for generator I/O. Default is 16.')
    parser.add_argument(
        '--train-csv',
        dest='train_csv',
        type=str,
        default=None,
        help='Custom path to training split CSV.')
    parser.add_argument(
        '--val-csv',
        dest='val_csv',
        type=str,
        default=None,
        help='Custom path to validation split CSV.')
    parser.add_argument(
        '--stats-dir',
        dest='stats_dir',
        type=str,
        default='saved_models',
        help='Directory containing channel_means.npy and channel_stds.npy.')
    return parser.parse_args()


def verify_environment(train_csv, val_csv, stats_dir):
    """Verifies existence of input split CSVs and precomputed statistics."""
    if not os.path.exists(train_csv):
        print(f"[ERROR] Training CSV path does not exist: {train_csv}")
        sys.exit(1)
    if not os.path.exists(val_csv):
        print(f"[ERROR] Validation CSV path does not exist: {val_csv}")
        sys.exit(1)

    means_p = os.path.join(stats_dir, 'channel_means.npy')
    stds_p = os.path.join(stats_dir, 'channel_stds.npy')
    if not (os.path.exists(means_p) and os.path.exists(stds_p)):
        print(f"[WARNING] Precomputed stats not found in '{stats_dir}'. Ensure precompute_stats.py was executed.")

    gpus = tf.config.list_physical_devices('GPU')
    print(f"--> Environment Check Passed | Available GPUs: {len(gpus)}")


def build_meaningful_cnn_lstm(input_shape=(21, 21, 137)):
    """
    Constructs a spatial CNN-LSTM architecture. Processes both row-wise and 
    column-wise spatial feature map sequences to remove directional bias across 
    the 21x21 meteorological patch.
    """
    inputs = layers.Input(shape=input_shape, name="Input_Patch")

    # Spatial Feature Extractor (Block 1)
    x = layers.Conv2D(64, (3, 3), padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # Shape: (10, 10, 64)

    # Spatial Feature Extractor (Block 2)
    x = layers.Conv2D(128, (3, 3), padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # Shape: (5, 5, 128)

    # 1. Row-wise Spatial Sequence: (Batch, Steps=5, Features=5*128=640)
    seq_rows = layers.Reshape((5, 5 * 128), name="Row_Spatial_Sequence")(x)
    lstm_rows = layers.LSTM(64, return_sequences=False,
                            kernel_regularizer=regularizers.l2(1e-4),
                            name="Row_LSTM")(seq_rows)

    # 2. Column-wise Spatial Sequence: Transpose (5, 5, 128) -> (5, 5, 128) along col dimension
    col_perm = layers.Permute((2, 1, 3), name="Transpose_Columns")(x)
    seq_cols = layers.Reshape((5, 5 * 128), name="Col_Spatial_Sequence")(col_perm)
    lstm_cols = layers.LSTM(64, return_sequences=False,
                            kernel_regularizer=regularizers.l2(1e-4),
                            name="Col_LSTM")(seq_cols)

    # Combine Dual-Axis Features (64 + 64 = 128 features)
    combined = layers.Concatenate(name="Dual_Axis_Merge")([lstm_rows, lstm_cols])
    combined = layers.Dropout(0.4)(combined)

    # Dense Classification Head
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Explicit FP32 output layer for Mixed Precision numerical stability
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32', name="Output_Probability")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CNN_LSTM_Meaningful_Model")
    return model


def main():
    args = parse_arguments()

    train_csv = args.train_csv or f"data/extracted_features_entire_world/tc_{args.leadtime}h_train.csv"
    val_csv   = args.val_csv or f"data/extracted_features_entire_world/tc_{args.leadtime}h_val.csv"

    verify_environment(train_csv, val_csv, args.stats_dir)

    log_dir = f"logs/cnn_lstm_{args.leadtime}h"
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_path = f'saved_models/best_cnn_lstm_{args.leadtime}h.keras'
    config_path     = f'saved_models/config_cnn_lstm_{args.leadtime}h.json'
    history_path    = f'saved_models/history_cnn_lstm_{args.leadtime}h.pkl'
    history_csv_p   = f'saved_models/history_cnn_lstm_{args.leadtime}h.csv'

    print(f"\n=================== TRAINING PIPELINE ({args.leadtime}h LEAD TIME) ===================")
    print(f"  • Model Type      : Dual-Axis Spatial CNN-LSTM")
    print(f"  • Train Label CSV : {train_csv}")
    print(f"  • Val Label CSV   : {val_csv}")
    print(f"  • Batch Size      : {args.batch_size}")
    print(f"  • Max Epochs      : {args.epochs}")
    print(f"  • CPU I/O Workers : {args.num_workers}")
    print(f"  • Checkpoint Path : {checkpoint_path}")
    print(f"  • TensorBoard Dir : {log_dir}")

    # Instantiate multi-threaded generators
    train_gen = TropicalCyclogenesisGenerator(
        train_csv,
        batch_size=args.batch_size,
        patch_size=21,
        neg_to_pos_ratio=3,
        is_train=True,
        enable_augmentation=True,
        stats_dir=args.stats_dir,
        num_workers=args.num_workers
    )

    val_gen = TropicalCyclogenesisGenerator(
        val_csv,
        batch_size=args.batch_size,
        patch_size=21,
        is_train=False,
        enable_augmentation=False,
        stats_dir=args.stats_dir,
        num_workers=args.num_workers
    )

    # Build dual-axis CNN-LSTM model
    model = build_meaningful_cnn_lstm(input_shape=(21, 21, 137))
    model.summary()

    # Compile with Binary Focal Crossentropy and Gradient Clipping
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.25),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc', curve='ROC'),
            tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    # Save training hyperparameter config metadata
    config_data = {
        "model_architecture": "Dual_Axis_CNN_LSTM",
        "leadtime": args.leadtime,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "neg_to_pos_ratio": 3,
        "patch_size": 21,
        "loss": "BinaryFocalCrossentropy",
        "gamma": 2.0,
        "alpha": 0.25,
        "seed": SEED,
        "optimizer": "Adam",
        "learning_rate": 1e-4,
        "clipnorm": 1.0,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "stats_dir": args.stats_dir,
        "tensorflow_version": tf.__version__,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=4)

    my_callbacks = [
        callbacks.EarlyStopping(
            monitor='val_pr_auc',
            mode='max',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_pr_auc',
            mode='max',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_pr_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    ]

    print("\n--> Starting Model Training Loop...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=my_callbacks,
        verbose=1
    )

    # Save history pickle and CSV logs
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)

    try:
        df_history = pd.DataFrame(history.history)
        df_history.to_csv(history_csv_p, index_label='epoch')
        print(f"  • History CSV Saved To : {history_csv_p}")
    except Exception as ex:
        print(f"  • Note: Could not dump history to CSV: {ex}")

    print(f"\n=================== TRAINING COMPLETE ===================")
    print(f"  • Best Model Saved To : {checkpoint_path}")
    print(f"  • History Saved To    : {history_path}")
    print(f"  • Config Saved To     : {config_path}")
    print(f"  • TensorBoard Logs    : {log_dir}")


if __name__ == '__main__':
    main()