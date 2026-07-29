#!/usr/bin/env python3

"""
Production Hybrid Conv-ViT-BiGRU Pipeline for Operational TC Genesis Detection.

Architecture Upgrades:
  1. Channel Attention (SE-Block): Dynamically reweights 137 input channels.
  2. Shallow CNN Feature Stem: Preserves local spatial/thermodynamic gradients.
  3. Pre-LN Transformer Stack (4 Blocks, d_model=192, heads=6, FFN=768).
  4. Spatial Sequence Bidirectional GRU (BiGRU) Aggregation.
  5. AdamW Optimizer (lr=3e-4, weight_decay=1e-4) with clipnorm=1.0.
  6. Standard Binary Cross-Entropy Loss for smooth recall convergence.
"""

import argparse
import json
import os
import pickle
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks, mixed_precision
from tcdl_generator import TropicalCyclogenesisGenerator

# Enforce Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Enable FP16 Mixed Precision Acceleration
try:
    mixed_precision.set_global_policy("mixed_float16")
    print("--> Mixed Precision policy set to 'mixed_float16' for GPU acceleration.")
except Exception as e:
    print(f"--> Could not set Mixed Precision policy: {e}")

tf.config.optimizer.set_experimental_options({"layout_optimizer": False})


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Hybrid Conv-ViT-BiGRU Tropical Cyclogenesis Model.")
    parser.add_argument('--leadtime', type=int, default=24, choices=[24, 48, 72])
    parser.add_argument('--batch-size', type=int, default=36)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--stats-dir', type=str, default='saved_models')
    return parser.parse_args()


class SqueezeAndExcitationBlock(layers.Layer):
    """Channel Attention SE-Block: Reweights 137 input variables based on importance."""
    def __init__(self, channels, reduction=8, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.global_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(channels // reduction, activation='relu', use_bias=False)
        self.fc2 = layers.Dense(channels, activation='sigmoid', use_bias=False)
        self.reshape = layers.Reshape((1, 1, channels))

    def call(self, inputs):
        se = self.global_pool(inputs)
        se = self.fc1(se)
        se = self.fc2(se)
        se = self.reshape(se)
        return inputs * se


class PatchExtractorAndEmbedder(layers.Layer):
    """Slices CNN feature maps into 3x3 spatial patches, flattens, and injects positional embeddings."""
    def __init__(self, patch_size=3, hidden_dim=192, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_patches = (21 // patch_size) ** 2
        
        self.projection = layers.Dense(hidden_dim, name="Patch_Projection")
        self.position_embeddings = layers.Embedding(
            input_dim=self.num_patches, output_dim=hidden_dim, name="Positional_Embedding"
        )

    def call(self, features):
        batch_size = tf.shape(features)[0]
        
        patches = tf.image.extract_patches(
            images=features,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        
        patch_dim = patches.shape[-1]
        patches = tf.reshape(patches, (batch_size, self.num_patches, patch_dim))
        
        embeddings = self.projection(patches)
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        pos_embeddings = self.position_embeddings(positions)
        
        return embeddings + pos_embeddings


def build_hybrid_vit_bigru(input_shape=(21, 21, 137), hidden_dim=192, num_heads=6, num_layers=4):
    """Constructs a Hybrid Channel-Attention CNN -> Pre-LN ViT -> BiGRU Architecture."""
    inputs = layers.Input(shape=input_shape, name="Input_Patch")
    
    # 1. Channel-Attention Stage (SE-Block)
    se_features = SqueezeAndExcitationBlock(channels=input_shape[-1], reduction=8, name="Channel_Attention")(inputs)
    
    # 2. Local Spatial CNN Feature Stem
    conv1 = layers.Conv2D(64, (3, 3), padding='same', name="Stem_Conv1")(se_features)
    conv1 = layers.BatchNormalization(name="Stem_BN1")(conv1)
    conv1 = layers.Activation('relu', name="Stem_ReLU1")(conv1)

    conv2 = layers.Conv2D(128, (3, 3), padding='same', name="Stem_Conv2")(conv1)
    conv2 = layers.BatchNormalization(name="Stem_BN2")(conv2)
    cnn_stem_output = layers.Activation('relu', name="Stem_ReLU2")(conv2)

    # 3. Patch Projection & Positional Embedding
    x = PatchExtractorAndEmbedder(patch_size=3, hidden_dim=hidden_dim, name="Patch_Embedder")(cnn_stem_output)
    
    # 4. Stacked Pre-LN Transformer Encoder Blocks (4 Blocks)
    for i in range(num_layers):
        norm_attn = layers.LayerNormalization(epsilon=1e-6, name=f"Transformer_Norm1_Block{i+1}")(x)
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim // num_heads,
            dropout=0.1,
            name=f"MultiHeadAttention_Block{i+1}"
        )(norm_attn, norm_attn)
        attn_output = layers.Dropout(0.1, name=f"Attn_Dropout_Block{i+1}")(attn_output)
        x = layers.Add(name=f"Add_Attn_Block{i+1}")([x, attn_output])
        
        # 4x FFN Expansion Ratio (192 -> 768 -> 192)
        norm_ffn = layers.LayerNormalization(epsilon=1e-6, name=f"Transformer_Norm2_Block{i+1}")(x)
        ffn = layers.Dense(hidden_dim * 4, activation='gelu', name=f"FFN_Expansion_Block{i+1}")(norm_ffn)
        ffn = layers.Dropout(0.1, name=f"FFN_Dropout_Block{i+1}")(ffn)
        ffn = layers.Dense(hidden_dim, name=f"FFN_Projection_Block{i+1}")(ffn)
        x = layers.Add(name=f"Add_FFN_Block{i+1}")([x, ffn])

    # 5. Spatial Sequence Processing via Bidirectional GRU (BiGRU)
    bigru_features = layers.Bidirectional(
        layers.GRU(128, return_sequences=False), name="Spatial_Sequence_BiGRU"
    )(x)
    bigru_features = layers.Dropout(0.25, name="BiGRU_Dropout")(bigru_features)
    
    # 6. Classification Head
    dense_out = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(bigru_features)
    dense_out = layers.BatchNormalization()(dense_out)
    dense_out = layers.Dropout(0.2)(dense_out)

    dense_out = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(dense_out)
    dense_out = layers.BatchNormalization()(dense_out)
    
    # FP32 output layer for Mixed Precision stability
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32', name="Output_Probability")(dense_out)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Hybrid_Conv_ViT_BiGRU_Model")
    return model


def main():
    args = parse_arguments()

    train_csv = f"data/extracted_features_entire_world/tc_{args.leadtime}h_train.csv"
    val_csv   = f"data/extracted_features_entire_world/tc_{args.leadtime}h_val.csv"

    print(f"\n=================== TRAINING PIPELINE ({args.leadtime}h LEAD TIME) ===================")
    print(f"  • Model Architecture : Hybrid Channel-Attention CNN + Pre-LN ViT + BiGRU")
    print(f"  • Train Label CSV    : {train_csv}")
    print(f"  • Val Label CSV      : {val_csv}")
    print(f"  • Batch Size         : {args.batch_size}")
    print(f"  • Max Epochs         : {args.epochs}")
    print(f"  • Process Workers    : {args.workers}")

    train_gen = TropicalCyclogenesisGenerator(
        train_csv,
        batch_size=args.batch_size,
        patch_size=21,
        neg_to_pos_ratio=3,
        is_train=True,
        enable_augmentation=True,
        stats_dir=args.stats_dir,
        num_workers=args.workers
    )

    val_gen = TropicalCyclogenesisGenerator(
        val_csv,
        batch_size=args.batch_size,
        patch_size=21,
        is_train=False,
        enable_augmentation=False,
        stats_dir=args.stats_dir,
        num_workers=args.workers
    )

    model = build_hybrid_vit_bigru(input_shape=(21, 21, 137), hidden_dim=192, num_heads=6, num_layers=4)
    model.summary()

    # Compile with AdamW, Binary Crossentropy, and PR-AUC monitoring
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4, clipnorm=1.0),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc', curve='ROC'),
            tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    os.makedirs('saved_models', exist_ok=True)
    log_dir = f'logs/vit_gru_{args.leadtime}h'
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_path = f'saved_models/best_vit_gru_{args.leadtime}h.keras'
    config_path     = f'saved_models/config_vit_gru_{args.leadtime}h.json'
    history_path    = f'saved_models/history_vit_gru_{args.leadtime}h.pkl'
    history_csv_p   = f'saved_models/history_vit_gru_{args.leadtime}h.csv'

    config_data = {
        "model_architecture": "Hybrid_Conv_ViT_BiGRU",
        "leadtime": args.leadtime,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "workers": args.workers,
        "hidden_dim": 192,
        "num_heads": 6,
        "num_layers": 4,
        "ffn_dim": 768,
        "loss": "BinaryCrossentropy",
        "seed": SEED,
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "clipnorm": 1.0
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=4)

    my_callbacks = [
        callbacks.EarlyStopping(
            monitor='val_pr_auc', mode='max', patience=10, restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_pr_auc', mode='max', factor=0.5, patience=4, min_lr=1e-6, verbose=1
        ),
        callbacks.ModelCheckpoint(
            checkpoint_path, monitor='val_pr_auc', mode='max', save_best_only=True, verbose=1
        ),
        callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

    print("\n--> Starting Hybrid ViT-BiGRU Model Training Loop...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=my_callbacks,
        verbose=1
    )

    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)

    try:
        import pandas as pd
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