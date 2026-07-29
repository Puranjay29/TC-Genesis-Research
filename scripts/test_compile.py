#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import layers, models
from train_ensemble import PatchExtractorAndEmbedder

def test_ensemble_build():
    print("Verifying structural layer extraction...")
    base_cnn = models.load_model('saved_models/best_cnn_lstm.keras')
    base_vit = models.load_model('saved_models/best_vit_gru.keras', custom_objects={'PatchExtractorAndEmbedder': PatchExtractorAndEmbedder})
    
    unified_input = layers.Input(shape=(21, 21, 137), name="Unified_Input_Grid")
    
    # Bypass Keras input node discovery bugs by extracting features straight from internal layer outputs
    cnn_features = base_cnn.get_layer("lstm").output
    vit_features = base_vit.get_layer("gru").output
    
    # Rebuild functional extractors using explicit references to base model inputs
    cnn_extractor = models.Model(inputs=base_cnn.inputs, outputs=cnn_features)
    vit_extractor = models.Model(inputs=base_vit.inputs, outputs=vit_features)
    
    out_cnn = cnn_extractor(unified_input)
    out_vit = vit_extractor(unified_input)
    
    fused_vector = layers.Concatenate(name="Feature_Fusion_Layer")([out_cnn, out_vit])
    x = layers.Dense(64, activation='relu')(fused_vector)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    ensemble_output = layers.Dense(1, activation='sigmoid', name="Ensemble_Probability")(x)
    ensemble_model = models.Model(inputs=unified_input, outputs=ensemble_output)
    
    ensemble_model.summary()
    print("\n✅ SUCCESS! Architecture compiled perfectly.")

if __name__ == '__main__':
    test_ensemble_build()
