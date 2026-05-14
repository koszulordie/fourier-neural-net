import click
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
import shap
import matplotlib.pyplot as plt
import seaborn as sns



# Early stopping callbacks
# ------------------------

early_stopping_patience = 5
early_stopping_monitor = 'val_loss'

early_stopping_callback = EarlyStopping(
    monitor = early_stopping_monitor,
    patience = early_stopping_patience,
    restore_best_weights = True,  # Crucial: ensures getting the model from the best epoch
    mode = 'min' if 'loss' in early_stopping_monitor else 'max' # Automatically sets mode based on metric
)


# Fourier Term
# ------------

@register_keras_serializable()
class FourierTerm(layers.Layer):
    def __init__(self, omega, n, **kwargs):
        super(FourierTerm, self).__init__(**kwargs)
        self.omega = omega
        self.n = n

    def get_config(self):
        config = super(FourierTerm, self).get_config()
        config.update({
            "omega": self.omega,
            "n": self.n,
        })
        return config

    def call(self, inputs):
        time_input, phase_shift = inputs
        return tf.sin(self.n * self.omega * time_input + phase_shift)


# Model wrapper
# -------------

class FourierModel:

    def __init__(self, input_dim, period, harmonics=[2], learning_rate=0.001, dropout_rate=0.1):
        
        hidden_layer_size = int((input_dim * 2 * len(harmonics)) ** 0.5)
        hidden_layers = [hidden_layer_size, hidden_layer_size]
        
        self.model = create_fourier_model(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,  # drop out for regularization
            period=period,  # e.g. 24.0 for a 24h period
            harmonics=harmonics  # list of harmonics
        )
        self.eval_results = None
        self.history = None
    
    def train(self, X_train_features, X_train_time, y_train, X_test_features, X_test_time, y_test):
        epochs=50
        batch_size=64
        self.history = self.model.fit(
            [X_train_features, X_train_time],
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,  # Use a portion of the resampled data for validation
            callbacks=[early_stopping_callback]
        )

        # Evaluate the model
        eval_results = self.model.evaluate([X_test_features, X_test_time], y_test, verbose=1)
        
        return eval_results
        
    def save(self, out_fn):
        self.model.save(out_fn)

    def plot_learning(self, out_fn):
        learning_curves(self.history, out_fn)



# Function to create model
# ------------------------

def create_fourier_model(input_dim, hidden_layers, period, harmonics, learning_rate=0.001, dropout_rate=0.0, verbose=False):
    
    # Define the inputs
    features_input = keras.Input(shape=(input_dim,), name='features_input')
    time_input = keras.Input(shape=(1,), name='time_input')

    # Build the hidden layers
    x = features_input
    for i, neurons in enumerate(hidden_layers):
        x = layers.Dense(neurons, activation='relu', name=f'hidden_layer_{i+1}')(x)
        x = layers.Dropout(dropout_rate)(x)

    # Compute angular frequency
    omega = 2 * np.pi / period

    # Add constant term a_0
    output = layers.Dense(1, use_bias=True, name='a0_term')(x)
    
    # Add harmonic terms
    for n in harmonics:

        # 1. Amplitude (A) based on features
        amplitude = layers.Dense(1, use_bias=False, name=f'amplitude_{n}')(x)
        
        # 2. Phase (phi) based on features
        # Note: No activation here allows phi to be any real number (radians)
        phase = layers.Dense(1, use_bias=False, name=f'phase_{n}')(x)

        # 3. Fourier layer = A * sin(n * omega * t + phase)
        # We pass both time and phase into the layer
        periodic_signal = FourierTerm(omega, n)([time_input, phase])
        harmonic_contribution = amplitude * periodic_signal
        
        output = layers.Add()([output, harmonic_contribution])

    # Create the model using the Functional API
    model = keras.Model(inputs=[features_input, time_input], outputs=output)
    
    if verbose:
        model.summary()

    print("\nCompiling the model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )

    return model


# Function to plot learning curves
# --------------------------------


def learning_curves(history, output_file):

    plt.figure(figsize=(4, 4))
    plt.plot(history.history['mean_absolute_error'], label='Train MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Val MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


@click.command()
@click.option('--data', type=click.Path(exists=True), required=True, help='Path to the input data file')
@click.option('--response', type=str, required=True, help='Response variable name')
@click.option('--out_data_noise', type=click.Path(), required=True, help='Path to save data with noise')
@click.option('--out_model_save', type=click.Path(), required=True, help='Path to save trained model')
@click.option('--out_learning_curve', type=click.Path(), required=True, help='Path to save learning curve plot')
def run(data, response, out_data_noise):

    df = pd.read_csv(data, sep='\t')
    df['noise'] = np.random.normal(0, 1, size=len(data))
    df.to_csv(out_data_noise, sep='\t', index=False)
    
    # format data

    covariates = ['sex', 'age', 'noise']
    X, y, time_data = preprocess_data(data, covariates, response, scale=False)

    # generate training data

    X_train_features, X_test_features, X_train_time, X_test_time, y_train, y_test = generate_training_data(X, y)

    # instantiate and train model

    model = FourierModel(X_train_features.shape[1], 24.0, harmonics=[2.,3.,4.,6.], learning_rate=0.005)
    model.train(X_train_features, X_train_time, y_train, X_test_features, X_test_time, y_test)
    model.save(out_model_save)

    # learning curves for diagnostics
    
    model.plot_learning(out_learning_curve)



    if __name__ == '__main__':
        
        run()
