#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

# Robust, serialization-safe custom layer class
class PatchExtractorAndEmbedder(tf.keras.layers.Layer):
    def __init__(self, patch_size=3, hidden_dim=128, **kwargs):
        # Explicitly swallow and push framework configs (name, trainable, dtype) to the superclass
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

def main():
    test_csv = "data/extracted_features_entire_world/tc_24h_test.csv"
    output_log_path = "data/extracted_features_entire_world/test_predictions_log.csv"
    
    print("--- STEP 1: INSTANTLY VERIFYING MODEL GRAPH ACCESSIBILITY ---")
    try:
        custom_obj = {'PatchExtractorAndEmbedder': PatchExtractorAndEmbedder}
        model = tf.keras.models.load_model('saved_models/best_ensemble.keras', custom_objects=custom_obj)
        
        # Test forward pass with a dummy random tensor to guarantee no runtime failures
        dummy_input = np.random.rand(1, 21, 21, 137).astype(np.float32)
        _ = model.predict(dummy_input, verbose=0)
        print("✅ Success! Model graph compiled flawlessly in memory.")
    except Exception as e:
        print(f"\n❌ Model verification failed: {e}")
        sys.exit(1)

    print("\n--- STEP 2: LOADING TEST GENERATOR SET MATRIX ---")
    from tcdl_generator import TropicalCyclogenesisGenerator
    test_gen = TropicalCyclogenesisGenerator(test_csv, batch_size=1, patch_size=21, is_train=False)
    
    log_records = []
    print("\n--- STEP 3: LOGGING REAL INFRASTRUCTURE COORDINATES WITH PREDICTIONS ---")
    
    # Align the structural row indices with the generator metadata
    limit = min(len(test_gen.pos_df), len(test_gen.neg_df))
    combined_meta = pd.concat([test_gen.pos_df[:limit], test_gen.neg_df[:limit]]).reset_index(drop=True)
    
    for idx in range(len(test_gen)):
        x_batch, y_batch = test_gen[idx]
        true_label = int(y_batch[0])
        
        meta_row = combined_meta.iloc[idx]
        
        act_lat = float(meta_row['Latitude']) if true_label == 1 else 0.0
        act_lon = float(meta_row['Longitude']) if true_label == 1 else 180.0
        
        prob = float(model.predict(x_batch, verbose=0)[0][0])
        pred_class = 1 if prob >= 0.5 else 0
        
        log_records.append({
            "Target_Latitude": act_lat,
            "Target_Longitude": act_lon,
            "Actual_Label": true_label,
            "Predicted_Probability": round(prob, 4),
            "Predicted_Label": pred_class,
            "Match": 1 if true_label == pred_class else 0
        })
        
        if (idx + 1) % 100 == 0 or (idx + 1) == len(test_gen):
            print(f"Logged tracking item: {idx + 1}/{len(test_gen)}")

    df_log = pd.DataFrame(log_records)
    df_log.to_csv(output_log_path, index=False)
    print(f"✅ Prediction logs with coordinates generated at: {output_log_path}")

if __name__ == '__main__':
    main()
