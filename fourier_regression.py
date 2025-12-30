import json
import click
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

from fourier_model import create_fourier_model, early_stopping_callback


def preprocess_data(df, covariates, response, scale=False):

    df = df[covariates + [response, 'time']].dropna()
    time_data = df['time'].values.reshape(-1, 1)  # reshape to 2D array

    stack = []
    for c in covariates:
        if scale:
            sc = StandardScaler()
            scaled = sc.fit_transform(df[c].values.reshape(-1, 1))
            stack.append(scaled)
        else:
            stack.append(df[c].values.reshape(-1, 1))
    
    X = np.hstack(stack + [time_data])
    y = df[response].values.reshape(-1, 1)
   
    return X, y, time_data


def generate_training_data(X, y, test_size=0.1):
    
    random_state = 123  # Random state for reproducibility
    input_dim = X.shape[1] - 1  # Number of features excluding time
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train_features = X_train[:, :input_dim]
    X_train_time = X_train[:, input_dim:]
    X_test_features = X_test[:, :input_dim]
    X_test_time = X_test[:, input_dim:]
    return X_train_features, X_test_features, X_train_time, X_test_time, y_train, y_test


def learning_curves(history, output_file):

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['mean_absolute_error'], label='Train MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Val MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_squared_error'], label='Train MSE')
    plt.plot(history.history['val_mean_squared_error'], label='Val MSE')
    plt.title('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_shap_absolute(shap_values, feature_names, output_file):

    # Plot mean absolute SHAP
    bars = np.mean(np.abs(shap_values), axis=0)
    plt.bar(x=range(len(bars)), height=bars[:,0])
    plt.xticks(range(len(bars)), feature_names, rotation=90)
    plt.ylabel('Mean Absolute SHAP')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_shap_heatmap(shap_values, y_train, index_selection, covariates, output_file):

    y_train_to_explain = y_train[index_selection]

    fig, axes = plt.subplots(1, 2, 
                             figsize=(10, 7),
                             gridspec_kw={
                                 'width_ratios': [5, 0.5],
                                 'wspace': 0.05})

    index_sorting = np.argsort(y_train_to_explain.reshape(-1))

    sns.heatmap(
        shap_values[index_sorting,:,0],
        ax=axes[0],
        xticklabels=covariates + ['time'], # Provide your list of column labels
        yticklabels=[],    # Provide your list of row labels
        cmap='bwr',        # Another good colormap choice
        annot=False,
        center=0,          # Show the values in each cell
        fmt=".1f",         # Format annotations to one decimal place
        # linewidths=.5,   # Add lines between cells
        # linecolor='white',   # Make lines white for contrast
        cbar=True,             # Display the color bar
        cbar_kws={'label': 'SHAP value'} # Label for the color bar
    )

    sns.heatmap(
        y_train_to_explain[index_sorting, :].reshape(-1, 1),
        ax=axes[1],               # Plot on the second subplot
        xticklabels=['response'], # Only one label for this column
        yticklabels=False,        # No y-tick labels for this subplot (already on the first)
        cmap='bwr',               # Custom black/white colormap
        annot=False,              # Show '0' or '1'
        fmt=".0f",                # Format as integers
        cbar=False,               # Add a color bar for binary (optional, can be False)
    )
    plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_partial_dependence_sex(model, X, output_file, response):

    n = 1000
    mean_age = np.mean(X[:, 1]) * np.ones((n, 1))
    X_sex_1 = np.ones((n, 1))
    X_sex_2 = 2 * np.zeros((n, 1))
    time = np.linspace(8, 20, num=n).reshape(-1, 1)
    X_1 = np.hstack([X_sex_1, mean_age])
    X_2 = np.hstack([X_sex_2, mean_age])
    
    # Get model predictions
    y_pred_1 = model.predict([X_1, time])
    y_pred_2 = model.predict([X_2, time])
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(time, y_pred_1, label='Sex: 1 (Male)')
    plt.plot(time, y_pred_2, label='Sex: 2 (Female)')
    plt.title('Partial Dependence Plot - Sex')
    plt.xlabel('Time')
    plt.ylabel(response)
    plt.legend()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_partial_dependence_age(model, output_file, response):

    n = 1000
    ages = [20, 30, 40, 50, 60]
    predicted = []
    X_mean_sex = np.zeros((n, 1))
    time = np.linspace(8, 20, num=n).reshape(-1, 1)
    for age in ages:
        X_age = age * np.ones((n, 1))
        X_stack_age = np.hstack([X_mean_sex, X_age])
        predicted.append(model.predict([X_stack_age, time]))

    for i, age in enumerate(ages):
        plt.plot(time, predicted[i], label=f'Predicted Age_{age}', lw=2, color=plt.cm.viridis(i / len(ages)))
    plt.xlabel('Time (hours)')
    plt.ylabel(response)
    plt.title('Predicted Values by Time')
    plt.legend()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


@click.command()
@click.option('--data', default='data/INTERVAL_for_ferran.xlsx', type=click.Path(), help='Input file path')
@click.option('--response', default='CRP_bl', help='Response variable to predict')
@click.option('--harmonics', default=3, help='Number of harmonics to use')
@click.option('--out_model', default='model.keras', type=click.Path(), help='Output model file name')
@click.option('--out_performance', default='learning_curves.png', type=click.Path(), help='Output performance file name')
@click.option('--out_shap_absolute', default='shap_values.png', type=click.Path(), help='Output average absolute SHAP')
@click.option('--out_shap_heatmap', default='shap_heatmap.png', type=click.Path(), help='Output SHAP heatmap file name')
@click.option('--out_partial_dependence_sex', default='partial_dependence_sex.png', type=click.Path(), help='Output partial dependence plot for sex')
@click.option('--out_partial_dependence_age', default='partial_dependence_age.png', type=click.Path(), help='Output partial dependence plot for age')
@click.option('--out_summary', default='summary.json', type=click.Path(), help='Output summary file name')
def main(data, response, harmonics, out_model, out_performance, out_shap_absolute, out_shap_heatmap, out_partial_dependence_sex, out_partial_dependence_age, out_summary):

    # Read input file
    dg = pd.read_csv(data, sep='\t')

    # Format data
    covariates = ['sex', 'age']
    X, y, time_data = preprocess_data(dg, covariates, response, scale=False)

    # Generate training data
    X_train_features, X_test_features, X_train_time, X_test_time, y_train, y_test = generate_training_data(X, y)

    # Instantiate model
    input_dim = len(covariates)
    hidden_layer_size = int(input_dim * 2 * harmonics)
    model = create_fourier_model(
        input_dim=X_train_features.shape[1],
        hidden_layers=[hidden_layer_size],
        learning_rate=0.001,
        dropout_rate=0.1,  # drop out for regularization
        period=24.0,  # 24h period
        n_harmonics=harmonics  # number of harmonics
    )

    # Train the model
    epochs=50
    batch_size=64
    history = model.fit(
        [X_train_features, X_train_time],
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,  # Use a portion of the resampled data for validation
        callbacks=[early_stopping_callback]
    )

    # Evaluate the model
    eval_results = model.evaluate([X_test_features, X_test_time], y_test, verbose=0)
    
    # Save the model
    model.save(out_model)

    # Plot learning curves
    learning_curves(history, out_performance)

    # SHAP

    # Define a wrapper function for the model's prediction
    # SHAP KernelExplainer requires a model function that takes a single numpy array
    # as input. Since our model takes two inputs, we combine them into a single array
    # and then separate them inside this wrapper function.
    # The `X` array passed to the wrapper will have shape (num_samples, input_dim + 1).
    def model_predict_wrapper(X):
        """
        Wrapper function for the model's prediction.
        It takes a single combined numpy array, separates it into features and time,
        and then calls the model's predict method.
        """
        X_features = X[:, :input_dim]
        X_time = X[:, input_dim:]

        # Return the model's prediction as a numpy array
        return model.predict([X_features, X_time])
    
    X_combined = np.hstack([X_train_features, X_train_time])
    # Select a background dataset for the explainer
    # A small, representative subset of the training data is typically used.
    background = X_combined[np.random.choice(X_combined.shape[0], 1000, replace=False)]
    print("\nInitializing SHAP KernelExplainer...")
    explainer = shap.KernelExplainer(model_predict_wrapper, background)

    # Select a random subset of data to explain
    
    num_samples_to_explain = 50
    index_selection = np.random.choice(X_combined.shape[0], num_samples_to_explain, replace=False)
    X_to_explain = X_combined[index_selection]

    # Compute the SHAP values
    print(f"Computing SHAP values for {num_samples_to_explain} samples... (This may take a moment)")
    shap_values = explainer.shap_values(X_to_explain)

    # Plot mean absolute SHAP
    plot_shap_absolute(shap_values, covariates + ['time'], out_shap_absolute)

    # Plot SHAP heatmap
    plot_shap_heatmap(shap_values, y_train, index_selection, covariates, out_shap_heatmap)

    # Partial dependence plots
    plot_partial_dependence_sex(model, X, out_partial_dependence_sex, response)
    plot_partial_dependence_age(model, out_partial_dependence_age, response)

    # Create summary
    summary = {}
    summary['input_file'] = data
    summary['response'] = response
    summary['harmonics'] = harmonics
    summary['eval_results'] = eval_results
    
    # summary['shap_values'] = shap_values
    # summary['partial_dependence_sex'] = out_partial_dependence_sex
    # summary['partial_dependence_age'] = out_partial_dependence_age

    # Save summary
    with open(out_summary, 'w') as f:
        json.dump(summary, f, indent=4)

if __name__ == '__main__':
    main()
