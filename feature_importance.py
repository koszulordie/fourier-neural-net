import click
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

from fourier_model import FourierTerm

from tensorflow.keras.models import load_model

from preprocess import preprocess_data, generate_training_data


class ShapImportance:

    def __init__(self, model, X_features, X_time, y, feature_names, response, nsamples_background=1000, nsamples=50):
        
        self.model = model
        self.X_features = X_features
        self.input_dim = X_features.shape[1]
        self.X_time = X_time
        self.y = y
        self.feature_names = feature_names
        self.nsamples_background = nsamples_background
        self.nsamples = nsamples
        self.response = response
        self.shap_values = None
        self.index_selection = None


    def compute_shap(self):

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
            X_features = X[:, :self.input_dim]
            X_time = X[:, self.input_dim:]

            # Return the model's prediction as a numpy array
            
            return self.model.predict([X_features, X_time])

        X_combined = np.hstack([self.X_features, self.X_time])

        # Select a background dataset for the explainer
        # A small, representative subset of the training data is typically used.

        background = X_combined[np.random.choice(X_combined.shape[0], self.nsamples_background, replace=False)]
        print("\nInitializing SHAP KernelExplainer...")
        explainer = shap.KernelExplainer(model_predict_wrapper, background)

        # Select a random subset of data to explain

        index_selection = np.random.choice(X_combined.shape[0], self.nsamples, replace=False)
        X_to_explain = X_combined[index_selection]

        # Compute the SHAP values
        print(f"Computing SHAP values for {self.nsamples} samples... (This may take a moment)")
        shap_values = explainer.shap_values(X_to_explain)

        self.shap_values = shap_values
        self.index_selection = index_selection


    def plot_shap_absolute(self, output_file):

        plt.figure(figsize=(5, 3)) 
        # Plot mean absolute SHAP
        bars = np.mean(np.abs(self.shap_values), axis=0)
        plt.bar(x=range(len(bars)), height=bars[:,0])
        plt.xticks(range(len(bars)), self.feature_names + ['time'], rotation=90)
        plt.ylabel('Mean Absolute SHAP')
        plt.title(f'Mean Absolute SHAP Values: {self.response}')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


    def plot_shap_boxplots(self, output_file):

        plt.figure(figsize=(5, 3))
        diff_test = {}
        # Plot mean absolute SHAP
        plt.boxplot(np.abs(self.shap_values[:,:,0]), tick_labels=self.feature_names + ['time'])
        plt.ylabel('SHAP')
        for i, covariate in enumerate(self.feature_names[:-1]):
            t_stat, p_val = stats.ttest_ind(np.abs(self.shap_values[:, i, 0]), np.abs(self.shap_values[:, -2, 0]), equal_var=False)
            mean_abs_diff = np.mean(np.abs(self.shap_values[:, i, 0])) - np.mean(np.abs(self.shap_values[:, -2, 0]))
            diff_test[covariate] = (mean_abs_diff, p_val)
        plt.title(f'Absolute SHAP Value Distribution: {self.response}')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        return diff_test

    def plot_shap_heatmap(self, output_file):

        y_to_explain = self.y[self.index_selection]

        fig, axes = plt.subplots(1, 2, 
                                figsize=(10, 7),
                                gridspec_kw={
                                    'width_ratios': [5, 0.5],
                                    'wspace': 0.05})

        index_sorting = np.argsort(y_to_explain.reshape(-1))

        sns.heatmap(
            self.shap_values[index_sorting,:,0],
            ax=axes[0],
            xticklabels=self.feature_names + ['time'], # Provide your list of column labels
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
            y_to_explain[index_sorting, :].reshape(-1, 1),
            ax=axes[1],               # Plot on the second subplot
            xticklabels=['response'], # Only one label for this column
            yticklabels=False,        # No y-tick labels for this subplot (already on the first)
            cmap='bwr',               # Custom black/white colormap
            annot=False,              # Show '0' or '1'
            fmt=".0f",                # Format as integers
            cbar=False,               # Add a color bar for binary (optional, can be False)
        )
        plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()



def plot_partial_dependence(model, feature_index, feature_grid, feature_names, X, response, output_file, n=1000):
    """
    model: Trained model
    feature_index: Index of the feature to vary
    feature_grid: Array of values for the feature to vary
    X: Original input data (to get means of other features)
    response: Name of the response variable (for labeling)
    output_file: File path to save the plot
    n: Number of points in the time grid
    """

    time_grid = np.linspace(8, 20, num=n).reshape(-1, 1)
    X_list = []
    for v in feature_grid:
        feature_array = np.ones((n, 1)) * v
        features = []
        for i in range(X.shape[1] - 1):
            if i == feature_index:
                features.append(feature_array)
            else:
                x = np.ones((n, 1)) * np.mean(X[:, i])
                features.append(x)
        X_list.append(np.hstack(features))
    
    # Get model predictions and plot
    
    plt.figure(figsize=(5, 3)) 

    # colormap to get different colors for each curve
    cmap = plt.get_cmap('copper')
    norm = plt.Normalize(vmin=min(feature_grid), vmax=max(feature_grid))

    for i, x in enumerate(X_list):
        y_pred = model.predict([x, time_grid])
        plt.plot(time_grid, y_pred, label=f'{feature_grid[i]:.2f}', color=cmap(norm(feature_grid[i])))
    plt.title(f'Partial Dependence Plot: {response} vs Feature {feature_names[feature_index]}')
    plt.xlabel('Time')
    plt.ylabel(response)
    plt.legend()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


@click.command()
@click.option('--data', type=click.Path(), help='Path to input data (pandas)')
@click.option('--model', type=click.Path(), help='Path to trained model')
@click.option('--response', default='CRP_bl', help='Response variable to predict')
@click.option('--out_partial_dependence_age', default='./tmp/partial_dependence_age.png', type=click.Path(), help='Output partial dependence plot for age')
@click.option('--out_partial_dependence_sex', default='./tmp/partial_dependence_sex.png', type=click.Path(), help='Output partial dependence plot for sex')
@click.option('--out_shap_absolute', default='./tmp/shap_mean_absolute.png', type=click.Path(), help='Output average absolute SHAP')
@click.option('--out_shap_boxplot', default='./tmp/shap_boxplots.png', type=click.Path(), help='Output SHAP boxplots')
def cli(model, data, response, out_partial_dependence_age, out_partial_dependence_sex, out_shap_absolute, out_shap_boxplot):

    # load model
    
    mod = load_model(model, custom_objects={"FourierTerm": FourierTerm})

    # get basic data

    covariates = ['sex', 'age', 'noise']
    df = pd.read_csv(data, sep='\t')
    X, y, time_data = preprocess_data(df, covariates, response, scale=False)
    X_train_features, X_test_features, X_train_time, X_test_time, y_train, y_test = generate_training_data(X, y)
    
    # Partial dependence plots
    # ------------------------

    plot_partial_dependence(mod, 0, [1, 2], ['sex', 'age', 'noise'], X, response, out_partial_dependence_sex)
    plot_partial_dependence(mod, 1, [40, 50, 60, 70], ['sex', 'age', 'noise'], X, response, out_partial_dependence_age)


    # SHAP
    # ----

    shap_importance = ShapImportance(mod, X_train_features, X_train_time, y_train, covariates, response, nsamples_background=1000, nsamples=50)
    shap_importance.compute_shap()


    # plot absolute SHAP contributions

    shap_importance.plot_shap_absolute(out_shap_absolute)

    # plot SHAP boxplots

    shap_importance.plot_shap_boxplots(out_shap_boxplot)


if __name__ == '__main__':
    
    cli()
