#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# Disable layout optimizer warning spam
tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

# 1. Redefine custom layer with full Keras serialization support (**kwargs) to absorb framework attributes
class PatchExtractorAndEmbedder(tf.keras.layers.Layer):
    def __init__(self, patch_size=3, hidden_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_patches = (21 // patch_size) ** 2
        self.projection = tf.keras.layers.Dense(hidden_dim)
        self.position_embeddings = tf.keras.layers.Embedding(input_dim=self.num_patches, output_dim=hidden_dim)

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
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

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size, "hidden_dim": self.hidden_dim})
        return config

def evaluate_model(model, test_gen):
    all_preds = []
    all_gt = []
    
    for i in range(len(test_gen)):
        x_batch, y_batch = test_gen[i]
        preds = model.predict(x_batch, verbose=0)
        all_preds.extend(preds.flatten())
        all_gt.extend(y_batch.flatten())
        
    y_true = np.array(all_gt)
    y_scores = np.array(all_preds)
    y_pred = (y_scores >= 0.5).astype(int)
    
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_scores),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0)
    }

def main():
    print("--- STEP 1: VERIFYING ALL MODELS LOAD CLEANLY FIRST ---")
    custom_obj = {'PatchExtractorAndEmbedder': PatchExtractorAndEmbedder}
    
    try:
        print("Checking CNN-LSTM...")
        model_cnn = tf.keras.models.load_model('saved_models/best_cnn_lstm.keras')
        print("Checking ViT-GRU...")
        model_vit = tf.keras.models.load_model('saved_models/best_vit_gru.keras', custom_objects=custom_obj)
        print("Checking Fused Ensemble...")
        model_ens = tf.keras.models.load_model('saved_models/best_ensemble.keras', custom_objects=custom_obj)
        print("✅ Success! All models compiled and ready in memory.")
    except Exception as e:
        print(f"\n❌ FAILED TO LOAD MODELS: {e}")
        print("Aborting script before loading data to save time.")
        sys.exit(1)

    print("\n--- STEP 2: INITIATING DATA PRELOAD LOOP ---")
    from tcdl_generator import TropicalCyclogenesisGenerator
    
    test_csv = "data/extracted_features_entire_world/tc_24h_test.csv"
    test_gen = TropicalCyclogenesisGenerator(test_csv, batch_size=32, patch_size=21, is_train=False)
    
    results = {}
    print("\nEvaluating CNN-LSTM...")
    results["CNN-LSTM Baseline"] = evaluate_model(model_cnn, test_gen)
    
    print("Evaluating ViT-GRU...")
    results["ViT-GRU Baseline"] = evaluate_model(model_vit, test_gen)
    
    print("Evaluating Fused Ensemble...")
    results["Fused Ensemble"] = evaluate_model(model_ens, test_gen)
    
    df_results = pd.DataFrame(results).T
    print("\n" + "="*24 + " FINAL BENCHMARK SUMMARY " + "="*24)
    print(df_results.to_string(formatters={
        "Accuracy": "{:.4f}".format, "AUC": "{:.4f}".format, 
        "Precision": "{:.4f}".format, "Recall": "{:.4f}".format, "F1-Score": "{:.4f}".format
    }))
    print("="*73)

if __name__ == '__main__':
    main()
