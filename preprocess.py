import json
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


@click.group()
def cli():
    pass


@cli.command()
@click.option('--input_file', type=click.Path(exists=True))
@click.option('--output_file', default='inputs.txt')
def preprocess(input_file, output_file):
    """
    python generate_inputs.py preprocess --input_file <input_file> --output_file <output_file>
    """

    # Read the input file
    data = pd.read_excel(input_file)
    data.rename(
        columns={
            'sexPulse': 'sex',
            'agePulse': 'age',
            'appointmentTime': 'time'
        },
        inplace=True
    )
    data['time'] = data['time'].astype(str)
    data['time'] = pd.to_timedelta(data['time']).dt.total_seconds() / (60 * 60)  # Convert to hours

    # Keep preprocessed data table
    data.to_csv(output_file, sep='\t', index=False)


@cli.command()
@click.option('--input_file', type=click.Path(exists=True))
@click.option('--output_file', type=click.Path(), default='responses.txt')
def responses(input_file, output_file, testing_mode=False):
    """
    python generate_inputs.py responses --input_file <input_file> --output_file <output_file>
    """

    # Read the input file
    data = pd.read_csv(input_file, sep='\t')
    df = {'response': []}
    count = 0
    for col in data.columns:
        if col not in ['identifier', 'sex', 'age', 'time']:
            if testing_mode and (count > 2):
                break
            df['response'].append(col)
            count += 1
    pd.DataFrame(df).to_csv(output_file, header=False, index=False)


if __name__ == '__main__':

    cli()
