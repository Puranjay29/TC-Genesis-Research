#!/usr/bin/env python3

"""
Production Probability-Aware 16-Token Cross-Attention Ensemble for TC Genesis.

Architectural Highlights:
  - Robust Name-Based Harvesting: Extract 320-dim feature vectors per model (LSTM/BiGRU + Dense).
  - Explicit Probability Branch: Integrates [p_cnn, p_vit, |diff|, mean, product] calibration vector.
  - 16-Token Sequence Projection: Splits features into 16 spatial tokens for Multi-Head Attention.
  - Learnable Residual Scale (Alpha=0.1) & Post-Interaction LayerNormalization.
  - Independent Scalar Confidence Gating & Compact MLP Head (128 -> 64 -> 1).
  - Two-Stage Training: Stage 1 Frozen Head Training -> Stage 2 End-to-End Fine-Tuning (lr=1e-6).
"""

import argparse
import json
import os
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers, mixed_precision, models, regularizers
from tcdl_generator import TropicalCyclogenesisGenerator

# Import custom layers for ViT deserialization
from train_vit_gru import PatchExtractorAndEmbedder, SqueezeAndExcitationBlock

# Enforce Exact Seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Enable Mixed Precision FP16
try:
    mixed_precision.set_global_policy("mixed_float16")
    print("--> Mixed Precision policy set to 'mixed_float16' for GPU acceleration.")
except Exception as e:
    print(f"--> Could not set Mixed Precision policy: {e}")

tf.config.optimizer.set_experimental_options({"layout_optimizer": False})


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train 16-Token Probability-Aware Gated Ensemble.")
    parser.add_argument('--leadtime', type=int, default=24, choices=[24, 48, 72])
    parser.add_argument('--batch-size', type=int, default=36)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--stats-dir', type=str, default='saved_models')
    return parser.parse_args()


class LayerScale(layers.Layer):
    """Learnable Residual Scaling factor initialized to alpha=0.1."""
    def __init__(self, init_value=0.1, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="layer_scale_gamma",
            shape=(input_shape[-1],),
            initializer=tf.keras.initializers.Constant(self.init_value),
            trainable=True
        )

    def call(self, x):
        return x * self.gamma


def harvest_cnn_features(cnn_model):
    """Harvests 320-dim features and probability output from CNN-LSTM via name lookup."""
    try:
        row_lstm = cnn_model.get_layer("Row_LSTM").output
        col_lstm = cnn_model.get_layer("Col_LSTM").output
        recurrent_feat = layers.Concatenate(name="CNN_Recurrent_Concat")([row_lstm, col_lstm]) # 128-dim
    except Exception:
        recurrent_feat = cnn_model.layers[13].output

    dense_layers = [l for l in cnn_model.layers if isinstance(l, layers.Dense)]
    dense_128 = dense_layers[-3].output  # 128-dim
    dense_64  = dense_layers[-2].output  # 64-dim
    prob_out  = cnn_model.get_layer("Output_Probability").output # 1-dim

    feat_320 = layers.Concatenate(name="CNN_Harvest_320")([recurrent_feat, dense_128, dense_64])
    return feat_320, prob_out


def harvest_vit_features(vit_model):
    """Harvests 320-dim features and probability output from Hybrid ViT-BiGRU via name lookup."""
    try:
        bigru_feat = vit_model.get_layer("Spatial_Sequence_BiGRU").output # 256-dim -> project to 128
        bigru_feat = layers.Dense(128, name="ViT_BiGRU_Proj")(bigru_feat)
    except Exception:
        bigru_feat = vit_model.layers[-7].output

    dense_layers = [l for l in vit_model.layers if isinstance(l, layers.Dense)]
    dense_128 = dense_layers[-3].output # 128-dim
    dense_64  = dense_layers[-2].output # 64-dim
    prob_out  = vit_model.get_layer("Output_Probability").output # 1-dim

    feat_320 = layers.Concatenate(name="ViT_Harvest_320")([bigru_feat, dense_128, dense_64])
    return feat_320, prob_out


def build_advanced_ensemble(cnn_lstm_path, vit_gru_path, input_shape=(21, 21, 137)):
    """Constructs a 16-Token Probability-Aware Gated Cross-Attention Ensemble Model."""
    print("--> Loading Pre-trained Expert Backbone Checkpoints...")
    custom_objects = {
        'SqueezeAndExcitationBlock': SqueezeAndExcitationBlock,
        'PatchExtractorAndEmbedder': PatchExtractorAndEmbedder
    }

    base_cnn = models.load_model(cnn_lstm_path, compile=False)
    base_vit = models.load_model(vit_gru_path, custom_objects=custom_objects, compile=False)

    base_cnn.trainable = False
    base_vit.trainable = False

    cnn_feat_320, cnn_prob = harvest_cnn_features(base_cnn)
    vit_feat_320, vit_prob = harvest_vit_features(base_vit)

    unified_input = layers.Input(shape=input_shape, name="Unified_Input_Grid")

    cnn_submodel = models.Model(inputs=base_cnn.inputs, outputs=[cnn_feat_320, cnn_prob], name="CNN_Harvester")
    vit_submodel = models.Model(inputs=base_vit.inputs, outputs=[vit_feat_320, vit_prob], name="ViT_Harvester")

    feat_cnn, p_cnn = cnn_submodel(unified_input)
    feat_vit, p_vit = vit_submodel(unified_input)

    # 1. Scalar Probability & Calibration Feature Vector (5-dim)
    p_diff = layers.Lambda(lambda x: tf.abs(x[0] - x[1]), name="Prob_Diff")([p_cnn, p_vit])
    p_mean = layers.Lambda(lambda x: (x[0] + x[1]) / 2.0, name="Prob_Mean")([p_cnn, p_vit])
    p_prod = layers.Multiply(name="Prob_Prod")([p_cnn, p_vit])
    prob_vector = layers.Concatenate(name="Probability_Calibration_Vector")([p_cnn, p_vit, p_diff, p_mean, p_prod])

    # 2. Projection to Unified Latent Space (320 -> 512)
    proj_cnn = layers.Dense(512, name="Proj_CNN_512")(feat_cnn)
    proj_cnn = layers.LayerNormalization(epsilon=1e-6, name="Norm_CNN_512")(proj_cnn)

    proj_vit = layers.Dense(512, name="Proj_ViT_512")(feat_vit)
    proj_vit = layers.LayerNormalization(epsilon=1e-6, name="Norm_ViT_512")(proj_vit)

    # 3. Independent Scalar Confidence Gating (1 Scalar per Model)
    gate_context = layers.Concatenate(name="Gate_Context")([prob_vector, layers.GlobalAveragePooling1D()(layers.Reshape((8, 64))(proj_cnn)), layers.GlobalAveragePooling1D()(layers.Reshape((8, 64))(proj_vit))])
    
    g_cnn_scalar = layers.Dense(1, activation="sigmoid", name="Scalar_Gate_CNN")(gate_context)
    g_vit_scalar = layers.Dense(1, activation="sigmoid", name="Scalar_Gate_ViT")(gate_context)

    gated_cnn = layers.Multiply(name="Gated_CNN_512")([proj_cnn, g_cnn_scalar])
    gated_vit = layers.Multiply(name="Gated_ViT_512")([proj_vit, g_vit_scalar])

    # 4. Construct 16-Token Sequence (8 CNN Tokens + 8 ViT Tokens)
    tokens_cnn = layers.Reshape((8, 64), name="Tokens_CNN")(gated_cnn)
    tokens_vit = layers.Reshape((8, 64), name="Tokens_ViT")(gated_vit)
    sequence_16_tokens = layers.Concatenate(axis=1, name="Tokens_16_Sequence")([tokens_cnn, tokens_vit])

    # 5. Multi-Head Cross-Attention over 16 Tokens
    attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1, name="Cross_Attention_16_Tokens")(
        sequence_16_tokens, sequence_16_tokens
    )
    attn_out = LayerScale(init_value=0.1, name="LayerScale_Attn")(attn_out)
    sequence_features = layers.LayerNormalization(epsilon=1e-6, name="Norm_Attn_Residual")(sequence_16_tokens + attn_out)

    pooled_tokens = layers.GlobalAveragePooling1D(name="Global_Token_Pool")(sequence_features) # 64-dim

    # 6. Interaction Vectors (Product & Absolute Difference) with LayerNormalization
    prod_vec = layers.Multiply(name="Interaction_Product")([proj_cnn, proj_vit])
    prod_vec = layers.LayerNormalization(epsilon=1e-6, name="Norm_Prod")(prod_vec)

    diff_raw = layers.Subtract(name="Interaction_Subtract")([proj_cnn, proj_vit])
    diff_vec = layers.Lambda(lambda x: tf.abs(x), name="Interaction_AbsDiff")(diff_raw)
    diff_vec = layers.LayerNormalization(epsilon=1e-6, name="Norm_Diff")(diff_vec)

    prod_pool = layers.Dense(64, activation='relu', name="Prod_Compress")(prod_vec)
    diff_pool = layers.Dense(64, activation='relu', name="Diff_Compress")(diff_vec)

    # 7. Final Master Fusion Vector
    master_fusion = layers.Concatenate(name="Master_Fusion_Vector")([
        pooled_tokens, prod_pool, diff_pool, prob_vector
    ])

    # 8. Compact MLP Classification Head
    x = layers.Dense(128, name="MLP_Dense128")(master_fusion)
    x = layers.BatchNormalization(name="MLP_BN1")(x)
    x = layers.Activation('gelu', name="MLP_GELU1")(x)
    x = layers.Dropout(0.2, name="MLP_Dropout1")(x)

    x = layers.Dense(64, name="MLP_Dense64")(x)
    x = layers.BatchNormalization(name="MLP_BN2")(x)
    x = layers.Activation('gelu', name="MLP_GELU2")(x)
    x = layers.Dropout(0.15, name="MLP_Dropout2")(x)

    outputs = layers.Dense(1, activation='sigmoid', dtype='float32', name="Ensemble_Probability")(x)

    model = models.Model(inputs=unified_input, outputs=outputs, name="Advanced_16Token_Gated_Ensemble")
    return model, base_cnn, base_vit


def main():
    args = parse_arguments()

    cnn_lstm_path = f'saved_models/best_cnn_lstm_{args.leadtime}h.keras'
    vit_gru_path  = f'saved_models/best_vit_gru_{args.leadtime}h.keras'

    if not os.path.exists(cnn_lstm_path) or not os.path.exists(vit_gru_path):
        raise FileNotFoundError("Required backbone checkpoints not found in saved_models/")

    train_csv = f"data/extracted_features_entire_world/tc_{args.leadtime}h_train.csv"
    val_csv   = f"data/extracted_features_entire_world/tc_{args.leadtime}h_val.csv"

    print(f"\n=================== TRAINING ADVANCED ENSEMBLE ({args.leadtime}h LEAD TIME) ===================")
    print(f"  • CNN Backbone : {cnn_lstm_path}")
    print(f"  • ViT Backbone : {vit_gru_path}")

    train_gen = TropicalCyclogenesisGenerator(
        train_csv, batch_size=args.batch_size, patch_size=21, neg_to_pos_ratio=3,
        is_train=True, enable_augmentation=True, stats_dir=args.stats_dir, num_workers=args.workers
    )

    val_gen = TropicalCyclogenesisGenerator(
        val_csv, batch_size=args.batch_size, patch_size=21,
        is_train=False, enable_augmentation=False, stats_dir=args.stats_dir, num_workers=args.workers
    )

    ensemble_model, base_cnn, base_vit = build_advanced_ensemble(cnn_lstm_path, vit_gru_path)
    ensemble_model.summary()

    # Stage 1: Train Fusion Head (Frozen Backbones)
    print("\n--> [STAGE 1/2] Training Ensemble Fusion Head (Frozen Backbones, lr=5e-5)...")
    ensemble_model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=1e-4, clipnorm=1.0),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.02),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc', curve='ROC'),
            tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    checkpoint_path = f'saved_models/best_ensemble_{args.leadtime}h.keras'
    config_path     = f'saved_models/config_ensemble_{args.leadtime}h.json'
    history_path    = f'saved_models/history_ensemble_{args.leadtime}h.pkl'

    callbacks_stage1 = [
        callbacks.EarlyStopping(monitor='val_pr_auc', mode='max', patience=6, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(checkpoint_path, monitor='val_pr_auc', mode='max', save_best_only=True, verbose=1)
    ]

    history1 = ensemble_model.fit(
        train_gen, validation_data=val_gen, epochs=15, callbacks=callbacks_stage1, verbose=1
    )

    # Stage 2: Unfreeze Top Dense Layers for Fine-Tuning
    print("\n--> [STAGE 2/2] Unfreezing Dense Layers for Fine-Tuning (lr=1e-6)...")
    base_cnn.trainable = True
    base_vit.trainable = True

    # Freeze lower conv/transformer bodies, leave upper dense heads trainable
    for layer in base_cnn.layers[:-4]:
        layer.trainable = False
    for layer in base_vit.layers[:-4]:
        layer.trainable = False

    ensemble_model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-6, weight_decay=1e-4, clipnorm=1.0),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.02),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc', curve='ROC'),
            tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    callbacks_stage2 = [
        callbacks.EarlyStopping(monitor='val_pr_auc', mode='max', patience=8, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(checkpoint_path, monitor='val_pr_auc', mode='max', save_best_only=True, verbose=1)
    ]

    history2 = ensemble_model.fit(
        train_gen, validation_data=val_gen, epochs=20, callbacks=callbacks_stage2, verbose=1
    )

    # Save History & Config
    with open(history_path, 'wb') as f:
        pickle.dump({'stage1': history1.history, 'stage2': history2.history}, f)

    config_data = {
        "model_architecture": "Advanced_16Token_Gated_Ensemble",
        "leadtime": args.leadtime,
        "batch_size": args.batch_size,
        "loss": "BinaryCrossentropy_LabelSmoothing_0.02",
        "stage1_lr": 5e-5,
        "stage2_lr": 1e-6,
        "seed": SEED
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=4)

    print(f"\n=================== ENSEMBLE TRAINING COMPLETE ===================")
    print(f"  • Best Model Checkpoint Saved To : {checkpoint_path}")
    print(f"  • Configuration Saved To          : {config_path}")


if __name__ == '__main__':
    main()