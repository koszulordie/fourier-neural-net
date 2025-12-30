import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
import shap
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping


@register_keras_serializable()
class FourierTerm(layers.Layer):
    def __init__(self, omega, n, **kwargs):
        super(FourierTerm, self).__init__(**kwargs)
        self.omega = omega
        self.n = n

    def call(self, time_input):
        cos_val = tf.cos(self.n * self.omega * time_input)
        sin_val = tf.sin(self.n * self.omega * time_input)
        return cos_val, sin_val

def create_fourier_model(input_dim, hidden_layers, period, n_harmonics, learning_rate=0.001, dropout_rate=0.0):
    
    # Define the inputs
    features_input = keras.Input(shape=(input_dim,), name='features_input')
    time_input = keras.Input(shape=(1,), name='time_input')

    # Build the hidden layers
    x = features_input
    for i, neurons in enumerate(hidden_layers):
        x = layers.Dense(neurons, activation='relu', name=f'hidden_layer_{i+1}')(x)
        x = layers.Dropout(dropout_rate)(x)

    # Calculate angular frequencies for harmonics
    omega = 2 * np.pi / period

    # Add constant term a_0
    a0 = layers.Dense(1, use_bias=True, name='a0_term')(x)
    output = a0

    # Add harmonic terms
    for n in range(1, n_harmonics + 1):
        # Create separate layers for sine and cosine coefficients
        an = layers.Dense(1, use_bias=False, name=f'an_term_{n}')
        bn = layers.Dense(1, use_bias=False, name=f'bn_term_{n}')

        # Use the custom FourierTerm layer to compute sin and cos terms
        cos_term_val, sin_term_val = FourierTerm(omega, n)(time_input)

        # Compute the final harmonic terms
        cos_term = an(x) * cos_term_val
        sin_term = bn(x) * sin_term_val
        
        output += cos_term + sin_term

    # Create the model using the Functional API
    model = keras.Model(inputs=[features_input, time_input], outputs=output)
    
    model.summary()

    print("\nCompiling the model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=[
            'mean_absolute_error',
            'mean_squared_error'
        ]
    )

    return model


# Early stopping callback

early_stopping_patience = 5
early_stopping_monitor = 'val_loss'

early_stopping_callback = EarlyStopping(
    monitor = early_stopping_monitor,
    patience = early_stopping_patience,
    restore_best_weights = True,  # Crucial: ensures you get the model from the best epoch
    mode = 'min' if 'loss' in early_stopping_monitor else 'max' # Automatically sets mode based on metric
)