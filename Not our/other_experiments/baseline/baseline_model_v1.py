# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# +
import sys  # noqa
sys.path.append('.')  # noqa
sys.path.append('..')  # noqa
sys.path.append('../../')  # noqa
sys.path.append('/home/puranjay/projects/TC-DL') # Absolute fallback path

import tc_formation.data.data as data
import tf_metrics as tfm
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
import plot
# -
# -

# Use model from [Matsuoka et. al](https://doi.org/10.1186/s40645-018-0245-y)

# Local data paths updated to match our split_dataset output layout
train_path = 'data/split_dataset/extracted_features_train'
val_path = 'data/split_dataset/extracted_features_val'
test_path = 'data/split_dataset/extracted_features_test'

# Exact dimensional footprint of your localized NetCDF slices: (lat, lon, channels)
data_shape = (41, 161, 137)

model = tf.keras.Sequential([
    # Input shape updated to match data_shape (41, 161, 137)
    layers.Conv2D(32, (3, 3), input_shape=(41, 161, 137)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(2048, activation='relu'),
    layers.Dense(2048, activation='relu'),
    layers.Dense(1),
])
model.summary()

# Build the model using BinaryCrossentropy loss

model.compile(
    optimizer='adam',
    # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True),
    metrics=[
        'binary_accuracy',
        tfm.RecallScore(from_logits=True),
        tfm.PrecisionScore(from_logits=True),
        tfm.F1Score(num_classes=1, from_logits=True, threshold=0.5),
    ]
)

# Load our training and validation data.

full_training = data.load_data(
    train_path,
    data_shape=data_shape,
    batch_size=64,
    shuffle=True,
)
downsampled_training = data.load_data(
    train_path,
    data_shape=data_shape,
    batch_size=64,
    shuffle=True,
    negative_samples_ratio=1)
validation = data.load_data(val_path, data_shape=data_shape)

normalizer = preprocessing.Normalization(axis=-1)
for X, y in iter(full_training):
    normalizer.adapt(X)
normalizer


# +
def normalize_data(x, y):
    return normalizer(x), y


full_training = full_training.map(normalize_data)
downsampled_training = downsampled_training.map(normalize_data)
validation = validation.map(normalize_data)
# -

# # First stage
#
# train the model on the down-sampled data.

# +
epochs = 50
first_stage_history = model.fit(
    downsampled_training,
    epochs=epochs,
    validation_data=validation,
    class_weight={1: 1., 0: 1.},
    shuffle=True,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            mode='max',
            verbose=1,
            patience=10,
            restore_best_weights=True),
    ]
)

plot.plot_training_history(first_stage_history, "First stage training")
# -

testing = data.load_data(test_path, data_shape=data_shape)
testing = testing.map(normalize_data)
model.evaluate(testing)

# # Second stage
#
# train the model on full dataset.

# +
second_stage_history = model.fit(
    full_training,
    epochs=epochs,
    validation_data=validation,
    class_weight={1: 10., 0: 1.},
    shuffle=True,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            mode='max',
            verbose=1,
            patience=10,
            restore_best_weights=True),
    ])


plot.plot_training_history(second_stage_history, "")
# -

# After the model is trained, we will test it on test data.

model.evaluate(testing)