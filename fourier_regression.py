import pickle
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from fourier_model import FourierModel
from preprocess import preprocess_data, generate_training_data
from feature_importance import ShapImportance, plot_partial_dependence


@click.command()
@click.option('--data', default='data/INTERVAL_for_ferran.xlsx', type=click.Path(), help='Input file path')
@click.option('--response', default='CRP_bl', help='Response variable to predict')
@click.option('--out_model', default='model.keras', type=click.Path(), help='Output model file name')
@click.option('--out_performance', default='learning_curves.png', type=click.Path(), help='Output performance file name')
@click.option('--out_shap_boxplots', default='shap_boxplots.png', type=click.Path(), help='Output SHAP boxplots file name')
@click.option('--out_partial_dependence_sex', default='partial_dependence_sex.png', type=click.Path(), help='Output partial dependence plot for sex')
@click.option('--out_partial_dependence_age', default='partial_dependence_age.png', type=click.Path(), help='Output partial dependence plot for age')
@click.option('--out_summary', default='summary.pickle', type=click.Path(), help='Output summary file name')
def main(data, response, out_model, out_performance, out_shap_boxplots, out_partial_dependence_sex, out_partial_dependence_age, out_summary):

    # MODEL INSTANCE AND TRAINING

    # Read input file
    dg = pd.read_csv(data, sep='\t')
    dg['noise'] = np.random.normal(0, 1, size=len(dg))  # Add noise feature for statistical testing

    # Format data
    covariates = ['sex', 'age', 'noise']
    X, y, time_data = preprocess_data(dg, covariates, response, scale=False)

    # Generate training data
    X_train_features, X_test_features, X_train_time, X_test_time, y_train, y_test = generate_training_data(X, y)

    # Initialize Fourier-valued NN model
    model = FourierModel(X_train_features.shape[1], 24.0, harmonics=[2.,3.,4.,6.], learning_rate=0.005)

    # Training and eval
    eval_results = model.train(X_train_features, X_train_time, y_train, X_test_features, X_test_time, y_test)

    # Save model instance
    model.save(out_model)

    # Plot learning curves
    model.plot_learning(out_performance)

    # FEATURE IMPORTANCE

    shap_importance = ShapImportance(model.model, X_train_features, X_train_time, y_train, covariates, response, nsamples_background=1000, nsamples=50)
    shap_importance.compute_shap()

    # mean absolute SHAP plot
    # shap_importance.plot_shap_absolute(out_shap_absolute)

    # SHAP absolute boxplot
    diff_test = shap_importance.plot_shap_boxplots(out_shap_boxplots)

    # SHAP heatmap plot
    # shap_importance.plot_shap_heatmap(out_shap_heatmap)

    # age-centric partial dependence
    plot_partial_dependence(model.model, 1, [40, 50, 60, 70], ['sex', 'age', 'noise'], X, response, out_partial_dependence_age)
    
    # sex-centric partial dependence
    plot_partial_dependence(model.model, 0, [1, 2], ['sex', 'age', 'noise'], X, response, out_partial_dependence_sex)

    # Summary object

    summary = {}
    summary['input_file'] = data
    summary['response'] = response
    summary['diff_test'] = diff_test
    summary['eval_results'] = eval_results
    summary['shap_values'] = shap_importance.shap_values
    summary['partial_dependence_sex'] = out_partial_dependence_sex
    summary['partial_dependence_age'] = out_partial_dependence_age

    with open(out_summary, 'wb') as f:
        pickle.dump(summary, f)

if __name__ == '__main__':
    main()
