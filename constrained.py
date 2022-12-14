from collections import Counter
from datetime import datetime
from imblearn.over_sampling import SMOTE
from keras import backend as K
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import tensorflow_addons as tfa

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf

# Data under license and not provided for this repository
indt = pd.read_json("data_cleaned.json")
# get target values
out = indt['labels'].values
indt.drop(['labels'], axis=1, inplace=True)

# Split into training (56%), testing (30%), validation (14%)
X_train, X_test, y_train, y_test = train_test_split(indt.values.astype("float32"),
                                                    out,
                                                    test_size=0.30,
                                                    random_state=42,
                                                    stratify=out)
X_train, X_val,  y_train, y_val  = train_test_split(X_train,
                                                    y_train,
                                                    test_size=0.20,
                                                    random_state=42,
                                                    stratify=y_train)

# Scaling expression data
scaler = MinMaxScaler()
X_train[:,5188:] = scaler.fit_transform(X_train[:,5188:])
X_test [:,5188:] = scaler.transform(X_test [:,5188:])
X_val  [:,5188:] = scaler.transform(X_val  [:,5188:])

# oversample training
X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)

# Convert to OHE
y_resampled = tf.keras.utils.to_categorical( y_resampled )
y_test      = tf.keras.utils.to_categorical( y_test )
y_val       = tf.keras.utils.to_categorical( y_val )

# Model definition
num_snps   = 5188
num_expr   = 19403
snps_input = keras.Input(shape=num_snps, name="snps")
expr_input = keras.Input(shape=num_expr, name="expr") 
# Add custom layer (non-trainable weights) here
pathways   = PathwayLayer(186, trainable=False, name="Pathways")(snps_input)
# SNP layers
resh       = keras.layers.Reshape((186,1))(pathways)
conv1D_s   = keras.layers.Conv1D(6,2,padding='same', activation=None,input_shape=[None,186,1])(resh)
flttn      = keras.layers.Flatten()(conv1D_s)
# Concatenate and batch normalize
concat = keras.layers.concatenate([ls1, le1])
bnorm  = keras.layers.BatchNormalization()(concat)
# hidden layers
l1 = layers.Dense(150,name="Layer1", 
                  kernel_regularizer=keras.regularizers.L1(1e-5),
                  activity_regularizer=keras.regularizers.L1(1e-4))(bnorm)
d1 = layers.Dropout(0.1)(l1) 
l2 = layers.Dense(30,name="Layer2", 
                  kernel_regularizer=keras.regularizers.L1(1e-5),
                  activity_regularizer=keras.regularizers.L1(1e-4))(d1)
d2 = layers.Dropout(0.1)(l2)
l3 = layers.Dense(15,name="Layer3", 
                  kernel_regularizer=keras.regularizers.L1(1e-5),
                  activity_regularizer=keras.regularizers.L1(1e-4))(d2)
# Output layer
lo         = layers.Dense(2,name="Output",activation="softmax")(l3)
# Model
model      = keras.Model(
                    inputs  = [snps_input, expr_input],
                    outputs = [lo],
)

# Learning schedule
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.001,
                    decay_steps=10,
                    decay_rate=0.96,
)
# Model build
model.compile(
                    optimizer= keras.optimizers.Adam(learning_rate=lr_schedule, name="adam"),
                    loss     = tf.keras.losses.BinaryCrossentropy(),
                    metrics  = [ 
                        tf.keras.metrics.CategoricalAccuracy(name='sp_acc'),
                        tf.keras.metrics.AUC(name='auc'),
                        tf.keras.metrics.FalsePositives(name='fp'),
                        tf.keras.metrics.Precision(name='pre'),
                        tf.keras.metrics.Recall(name='rec'),
                    ]
)
# Training
callback  = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=8, verbose=1, mode="min", restore_best_weights=True)
history   = model.fit(
                    [X_resampled[:,:num_snps], X_resampled[:,num_snps:]], 
                    y_resampled, 
                    epochs         = 120,
                    callbacks      = callback,
                    validation_data= ([X_val[:,:num_snps], X_val[:,num_snps:]], y_val), 
                    verbose        = False,
)

# Evaluate on test set
model.evaluate([X_test[:,:num_snps], X_test[:,num_snps:]],y_test)

# SHAP feature importance
select = np.random.randint( len(X_train), size=100 )
# select backgroud for shap
background = [X_train[select,:num_snps], X_train[select,num_snps:]]
# DeepExplainer to explain predictions of the model
explainer = shap.DeepExplainer(model, background)
# compute shap values
shap_values = explainer.shap_values([X_test[:,:num_snps], X_test[:,num_snps:]], check_additivity=True)
