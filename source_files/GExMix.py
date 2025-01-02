import matplotlib
matplotlib.use('Agg')  # do not show plots while running code. Use "TKAgg" if you want to see figure right away
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from itertools import islice
from os import path, makedirs, listdir
from sklearn.svm import SVC
from datetime import datetime
from glob import glob

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import zscore, spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import ElasticNet, ElasticNetCV, MultiTaskElasticNetCV, Lasso, \
        LinearRegression, BayesianRidge, TweedieRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from textwrap import wrap


# Get the path of the current directory (source_files)
current_dir = path.dirname(path.abspath(__file__))
# Construct the path to the project directory
project_dir = path.dirname(current_dir)
# Construct the path to the modules directory
module_dir = path.join(project_dir, 'source_files')
# Add the module directory to sys.path
sys.path.append(module_dir)

from plotting_utils import (load_parameters, get_reduced_dimension_projection,
                            visualize_data_with_meta, interactive_bokeh_with_select_test_for_DGEA)

from Datasets import (GExDataset, TCGAClinicalDataProcessor, GExMix, DatasetVisualizer, GDSCDataProcessor,
                      GDSCModelCollection, get_filename_dict)


def visualize_augmentation(feature_list, label_list, name_list, prefix="BRCA", save_dir="./"):
    """
    compute projection of a list of feature and labels
    :param feature_list:
    :param label_list:
    :param save_dir:
    :return:
    """
    concat_features = np.concatenate(feature_list, axis=0)
    projection = get_reduced_dimension_projection(concat_features, vis_mode="umap",
                                                           n_components=2,
                                                           n_neighbors=30)
    # Define markers and colors for different labels
    markers = ['*', 'd', 'o', 's', '^', 'x']

    fig, ax = plt.subplots()
    start_idx = 0
    # Plot each feature set with a different marker
    for i, (feature, labels, name) in enumerate(zip(feature_list, label_list, name_list)):
        end_idx = start_idx + len(labels)
        marker = markers[i % len(markers)]
        ax.scatter(projection[start_idx:end_idx, 0], projection[start_idx:end_idx, 1], c=labels,
                   marker=marker, label=name, cmap=plt.cm.viridis)
    plt.title(prefix)
    # Add colorbar
    cb = plt.colorbar(ax.collections[0], ax=ax)
    cb.set_label('Tumor purity')
    # Add legend and labels
    ax.legend()
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.savefig(path.join(save_dir, f"{prefix}_augment_visualization_.png"))
    plt.close()

    # plot original and augment separately
    fig, axes = plt.subplots(1, 2, figsize=[10, 5])

    # Initialize starting index
    sc = axes[0].scatter(projection[:len(label_list[0]), 0], projection[:len(label_list[0]), 1],
                         c=label_list[0], marker=markers[0], label=name_list[0], cmap="viridis",
                         s=10)
    axes[0].legend()
    axes[0].set_title(prefix)
    cb = fig.colorbar(sc, ax=axes[0])
    cb.set_label('Tumor purity')

    # Initialize starting index
    sc = axes[1].scatter(projection[len(label_list[0]):, 0], projection[len(label_list[0]):, 1],
                         c=label_list[1], marker=markers[1], label=name_list[1], cmap="viridis",
                         s=10)
    axes[1].legend()
    cb = fig.colorbar(sc, ax=axes[1])
    cb.set_label('Tumor purity')
    plt.savefig(path.join(save_dir, f"{prefix}_2in1_augment.png"))
    plt.close()


def round_to_closest_target(predictions, targets):
    rounded_predictions = []
    for pred in predictions:
        closest_target = targets[np.argmin(np.abs(targets - pred))]
        rounded_predictions.append(closest_target)
    return np.array(rounded_predictions)


def focal_loss_regression(alpha=1.0, gamma=2.0, base_loss="mse"):
    """
    Focal loss for regression tasks, emphasizing hard examples based on residuals.

    Args:
        alpha (float): Scaling factor for the loss.
        gamma (float): Focusing parameter to emphasize harder examples.
        base_loss (str): Base loss function ('mse' or 'mae').

    Returns:
        A callable loss function.
    """

    def focal_loss(y_true, y_pred):
        # Compute residuals
        residual = tf.abs(y_true - y_pred)  # Absolute error
        base = tf.keras.losses.mean_squared_error if base_loss == "mse" else tf.keras.losses.mean_absolute_error

        # Compute base loss
        base_loss_value = base(y_true, y_pred)

        # Focal term: Scale by residual error
        focal_term = tf.pow(residual + tf.keras.backend.epsilon(), gamma)

        # Combine focal term and base loss
        loss = alpha * focal_term * base_loss_value

        # Reduce mean over batch
        return tf.reduce_mean(loss)

    return focal_loss


def get_regression_model_given_name(model_name, input_shape=400, gamma=2.0, alpha=0.25,
                         droprate=0.58, class_weight={0: 0.5, 1: 0.5}, **kwargs):

    if model_name.lower() == "bayesianridge":
        model = BayesianRidge()
    elif model_name.lower() == "lasso":
        model = Lasso(
            alpha=1.0,
            fit_intercept=True,
            normalize="deprecated",
            precompute=False,
            copy_X=True,
            max_iter=10000,
            tol=1e-4,
            warm_start=False,
            positive=False,
            random_state=899,
            selection="cyclic")
    elif model_name.lower() == "elasticnet":
        model = ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                fit_intercept=True,
                max_iter=10000,
                random_state=99,
                selection="cyclic")
    elif model_name.lower() == "linearregression":
        model = LinearRegression()
    elif model_name.lower() == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.05,
            eval_metric="logloss", subsample=0.7, colsample_bylevel=0.7,
            min_child_weight=1)
    elif model_name.lower() == "tweedieregressor":
        model = TweedieRegressor(power=1, alpha=0.5, link='log')
    elif model_name.lower() == "fnn":
        model = Sequential(
                [
                        Dense(128, input_dim=input_shape, activation='selu'),
                        BatchNormalization(),
                        Dropout(0.4),
                        Dense(64, activation='selu'),
                        BatchNormalization(),
                        Dropout(0.4),
                        Dense(1, activation="sigmoid")
                        # Use sigmoid for multi-label classification
                ])

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001), loss='mse',
            metrics=['accuracy'])
        # Compile model with focal loss
        # model.compile(
        #         optimizer=Adam(learning_rate=0.0005),
        #         loss=focal_loss_regression(alpha=0.25, gamma=2.0),
        #         metrics=['accuracy']
        # )

    return model


def get_classifier_model_given_name(model_name, input_shape=400, output_shape=3, gamma=2.0, alpha=0.25,
                                    droprate=0.58, class_weight={0: 0.5, 1: 0.5}, **kwargs):
    if model_name.lower() == "svm":
        model = SVC(probability=True, class_weight=class_weight)
    elif model_name.lower() == "lasso":
        model = Lasso(
            alpha=1.0,
            fit_intercept=True,
            normalize="deprecated",
            precompute=False,
            copy_X=True,
            max_iter=10000,
            tol=1e-4,
            warm_start=False,
            positive=False,
            random_state=899,
            selection="cyclic")
    elif model_name.lower() == "rf":
        model = RandomForestClassifier(max_depth=50, random_state=99)
    elif model_name.lower() == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.05,
            eval_metric="logloss", subsample=0.7, colsample_bylevel=0.7,
            min_child_weight=1)
    elif model_name.lower() == "elasticNet":
        model = ElasticNet(
            alpha=0.9,
            l1_ratio=0.5,
            fit_intercept=True,
            normalize="deprecated",
            precompute="auto",
            max_iter=10000,
            random_state=99,
            selection="cyclic")
    elif model_name.lower() == "sk_mlp":
        model = MLPClassifier(
            hidden_layer_sizes=32, activation="relu", validation_fraction=0.2, max_iter=10000)
    elif model_name.lower() == "regression":
        model = LinearRegression()
    elif model_name.lower() == "knn":
        model = KNeighborsClassifier()
    elif model_name.lower() == "keras_fnn":
        import tensorflow.keras.layers as layers
        from tensorflow.keras import Input, Model, optimizers
        ## combine multiple inputs together

        inputs = Input(shape=(input_shape,))  # shape: num_samples * 20,000
        dense1 = layers.Dense(128, activation="selu")(inputs)
        dense1 = layers.Dropout(droprate)(dense1)
        dense1 = layers.Dense(32, activation="selu")(dense1)
        dense1 = layers.Dropout(droprate)(dense1)
        out = layers.Dense(output_shape, activation="softmax")(dense1)

        # train model
        model = Model(inputs=inputs, outputs=[out])
        opt = optimizers.Adam(learning_rate=0.001)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        # model.compile(
        #         optimizer=opt, loss=focal_loss(gamma=gamma, alpha=alpha), metrics=['accuracy']
        # )

    elif model_name.lower() == "keras_cnn":
        import tensorflow.keras.layers as layers
        from tensorflow.keras import Input, Model, optimizers
        ## cat C
        inputs = Input(shape=(input_shape, 1))
        x = layers.Conv1D(16, 3, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.AlphaDropout(droprate)(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv1D(32, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.AlphaDropout(droprate)(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv1D(64, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.AlphaDropout(droprate)(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv1D(128, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.AlphaDropout(droprate)(x)
        x = layers.Activation("relu")(x)
        gap_x = layers.GlobalAveragePooling1D()(x)

        out = layers.Dense(output_shape, activation="softmax")(gap_x)

        # train model
        model = Model(inputs=inputs, outputs=[out])
        opt = optimizers.Adam(learning_rate=0.0005)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        # model.compile(
        #         optimizer=opt, loss=focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy']
        # )

        model.summary()

    return model

def encode_labels(df, column_name):
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(df[column_name])
    return encoded_labels, encoder.classes_v


## 2024.11.18
def train_evaluate_and_log(model_name, X_train, X_val, y_train_df, tcga_y_test_df, overall_results_df,
                           tcga_X_test=None, y_val_df=None, num_epochs=50, run=0, drug="drug", y_target_col="binary_response",
                           ori_or_aug="aug", save_dir="./", aug_by_col="aug_by_col",
                           run_postfix="run_postfix", title_suffix="",
                           regress_or_classify="regress"):
    """Train, evaluate, and log results for a model."""
    if isinstance(y_train_df, pd.DataFrame):
        # filter out binary response
        y_train_values = y_train_df[~y_train_df[y_target_col].isna()][y_target_col]
        X_train_values = X_train[~y_train_df[y_target_col].isna()]
    elif isinstance(y_train_df, np.ndarray):
        y_train_values = y_train_df
        X_train_values = X_train
    if isinstance(y_val_df, pd.DataFrame):
        y_val_values = y_val_df[~y_val_df[y_target_col].isna()][y_target_col]
        X_val_values = X_val[~y_val_df[y_target_col].isna()]
    elif isinstance(y_val_df, np.ndarray):
        y_val_values = y_val_df
        X_val_values = X_val
    # Train the model
    history, model = train_and_evaluate_model(
            model_name, X_train_values, X_val_values, y_train_values, y_val_values,
            num_epochs, regress_or_classify=regress_or_classify
    )

    # Plot training history
    if history is not None:
        plot_history_train_val(
            history, title=f"{model_name} {ori_or_aug} data training {title_suffix}",
            postfix=run_postfix + f"-{model_name} {ori_or_aug}", save_dir=save_dir
        )

    # Evaluate the model
    gt_predict_df = pd.DataFrame(columns=["prediction", "ground_truth"])
    if tcga_X_test is not None:
        y_pred, accuracy = evaluate_model(model, tcga_X_test, tcga_y_test_df[y_target_col].values)
        gt_predict_df["ground_truth"] = tcga_y_test_df[y_target_col].values
    else:
        y_pred, accuracy = evaluate_model(model, X_val, y_val_df[y_target_col].values)
        gt_predict_df["ground_truth"] = y_val_df[y_target_col].values

    y_pred = y_pred.flatten()
    gt_predict_df["prediction"] = y_pred

    ## compute metrics
    # Drop rows with NaN values in either 'ground_truth' or 'prediction'
    gt_predict_df_filtered = gt_predict_df.dropna(subset=["ground_truth", "prediction"])
    if len(gt_predict_df_filtered) >= 2:
        pearsonr_corr, _ = pearsonr(gt_predict_df_filtered["ground_truth"],
                                    gt_predict_df_filtered["prediction"])
        sp_corr, p_value = spearmanr(gt_predict_df_filtered["ground_truth"], gt_predict_df_filtered["prediction"])
        r2 = r2_score(gt_predict_df_filtered["ground_truth"], gt_predict_df_filtered["prediction"])
    else:
        pearsonr_corr = None
        sp_corr = None
        r2 = None

    # Append results to DataFrame
    overall_results_df = pd.concat(
        [
            overall_results_df,
            pd.DataFrame(
                [{
                    "drug": drug,
                    "ori_or_aug": ori_or_aug,
                    "model_name": model_name,
                    "run": run,
                    "accuracy": accuracy,
                    "pearson_score": pearsonr_corr,
                    "spearman_score": sp_corr,
                    "r2_score": r2,
                    "num_samples": len(X_train),
                    "aug_by_col": aug_by_col,
                }]
            )
        ], ignore_index=True
    )
    # Plot prediction vs ground truth
    plot_prediction_gt_scatter(
        y_pred, tcga_y_test_df, y_target_col=y_target_col, postfix=f"{model_name} {ori_or_aug}_{run_postfix}",
        title=f"Test set Accuracy: {accuracy:.2f} ({model_name} {ori_or_aug}) {title_suffix}",
        save_dir=save_dir
    )

    if len(gt_predict_df["ground_truth"]) >= 2:
        # Filter for rows with non-NaN values in both columns
        filtered_df = gt_predict_df.dropna(subset=["ground_truth", "prediction"])
        try:
            if not filtered_df.empty and len(filtered_df) > 1:
                # Compute metrics
                pearsonr_corr, _ = pearsonr(filtered_df["ground_truth"], filtered_df["prediction"])
                sp_corr, p_value = spearmanr(filtered_df["ground_truth"], filtered_df["prediction"])
                r2 = r2_score(filtered_df["ground_truth"], filtered_df["prediction"])
            else:
                # Handle case with no valid data
                print("No valid data available for computations.")
                pearsonr_corr, sp_corr, p_value, r2 = 0, 0, 0, 0
        except ValueError as e:
            print(f"Error computing metrics: {e}")
            pearsonr_corr, sp_corr, p_value, r2 = 0, 0, 0, 0
        corr_str = f"Pearson r: {pearsonr_corr:.2f}\nSpearman r: {sp_corr:.2f}\nR2:{r2:.2f}"

        plot_violin_grouped_by_metric(
            data_df=pd.concat([tcga_y_test_df, gt_predict_df], axis=1),
            group_by_col="diagnosis",
            metric1_col="ground_truth",
            metric2_col="prediction",
            metric1_label="Ground Truth",
            metric2_label="Prediction",
            prefix=f"{model_name}-{ori_or_aug}-{run_postfix}",
            corr_str=corr_str,
            save_dir=save_dir,
            plot_mode="violin"
        )

    return overall_results_df, gt_predict_df


def load_saved_drug_response_get_visualization():
    # Combine datasets for easier analysis
    filename = r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active projects\1-pancan_subtyping\results\GDSC-RES\GDSC-aug-with-TCGA\12-8T11-39-OnlyTcga-randAug\12-08T11-39-overall_drp_vinorelbine.csv"
    save_dir2 = path.dirname(filename)
    loaded_all_drug_data = pd.read_csv(filename)

    for model_name in loaded_all_drug_data["model_name"].unique():
        fnn_df = loaded_all_drug_data[loaded_all_drug_data["model_name"] == model_name]
        # Group by drug and original/augment to calculate mean accuracy
        grouped_data = fnn_df.groupby(['drug', 'original_or_augment'])['accuracy'].agg(
                ['mean', 'std']).reset_index()
        # Add improvement (augment - original)
        original_acc = grouped_data[grouped_data['original_or_augment'] == 'original']
        augment_acc = grouped_data[grouped_data['original_or_augment'] == 'augment']
        improvement = augment_acc.set_index('drug')['mean'] - original_acc.set_index('drug')['mean']
        grouped_data['improvement'] = grouped_data['drug'].map(improvement)
        # Create a barplot with error bars
        plt.figure(figsize=(14, 8))
        sns.barplot(
                data=grouped_data,
                x='drug',
                y='mean',
                hue='original_or_augment',
                palette='pastel',
                ci="sd",
        )
        # Add error bars
        for index, row in grouped_data.iterrows():
            plt.errorbar(
                    x=index // 2 + (0.2 if row['original_or_augment'] == 'original' else -0.2),
                    y=row['mean'],
                    yerr=row['std'],
                    fmt='none',
                    capsize=5,
                    color='black'
            )
        plt.title(f'Mean Accuracy per drug ({model_name})\nTCGA -> TCGA')
        plt.ylabel('Mean Accuracy')
        plt.xlabel('Drug')
        plt.xticks(rotation=45, ha="right")
        plt.legend(title='Data Type')
        plt.tight_layout()
        plt.savefig(fr"{save_dir2}\{model_name}mean_accuracy_by_drug_and_data_type.png")
    plt.show()

def plot_violin_grouped_by_metric(
        data_df, group_by_col, metric1_col, metric2_col, metric1_label, metric2_label,
        prefix, corr_str=None, save_dir="./", plot_title=None, plot_mode="box"
):
    """
    Creates a split violin plot grouped by a specified column, comparing two metrics.

    Args:
        data_df (pd.DataFrame): DataFrame containing the data.
        group_by_col (str): Column name for grouping values (e.g., "TCGA Classification").
        metric1_col (str): Column name for the first metric (e.g., ground truth).
        metric2_col (str): Column name for the second metric (e.g., prediction).
        metric1_label (str): Label for the first metric (e.g., "Ground Truth").
        metric2_label (str): Label for the second metric (e.g., "Prediction").
        prefix (str): Prefix for plot titles and saved file names.
        corr_str (str): Text to display correlation information on the plot.
        save_dir (str): Directory to save the plot.
        plot_title (str): Title for the plot. If None, a default title will be generated.
    """
    # Filter for rows with non-NaN values in both columns
    data_df = data_df.dropna(subset=["ground_truth", "prediction"])

    # Prepare data for violin plot
    metric1_values = data_df[[group_by_col, metric1_col]].rename(columns={metric1_col: 'value'})
    metric1_values['value_type'] = metric1_label

    metric2_values = data_df[[group_by_col, metric2_col]].rename(columns={metric2_col: 'value'})
    metric2_values['value_type'] = metric2_label

    # Combine data
    extended_df = pd.concat([metric1_values, metric2_values], ignore_index=True)
    extended_df[group_by_col] = extended_df[group_by_col].astype(
        str)  # in case of "nan" classes

    # Compute Spearman correlation coefficient for each group
    spearman_results = {}
    for group in extended_df[group_by_col].unique():
        group_data = extended_df[extended_df[group_by_col] == group]
        metric1_group = group_data[group_data['value_type'] == metric1_label]['value']
        metric2_group = group_data[group_data['value_type'] == metric2_label]['value']
        if len(metric1_group) > 1 and len(metric2_group) > 1:  # Ensure sufficient data points
            rho, p_value = spearmanr(metric1_group, metric2_group)
            spearman_results[group] = (rho, p_value)
        else:
            spearman_results[group] = (None, None)
    min_cut = np.min(extended_df['value'])
    # Create the violin plot
    if plot_mode == "violin":
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(
                x=group_by_col, y='value', hue='value_type', data=extended_df,
                split=True, palette='pastel', width=1, inner="quart", alpha=0.99
        )
        sns.stripplot(
                x=group_by_col, y='value', hue='value_type', data=extended_df,
                size=6, jitter=True, alpha=0.7
        )
    elif plot_mode == "box":
        plt.figure(figsize=(12, 6))
        # Add alternating gray background shades
        unique_classes = extended_df[group_by_col].unique()
        for i, cls in enumerate(unique_classes):
            if i % 2 == 0:  # Apply shading to alternate classes
                plt.axvspan(i - 0.5, i + 0.5, color='lightgray', alpha=0.3, zorder=0)
        ax = sns.boxplot(
                x=group_by_col, y='value', hue='value_type', data=extended_df,
                palette='pastel'
        )
        sns.stripplot(
                x=group_by_col, y='value', hue='value_type', data=extended_df,
                size=6, jitter=True, alpha=0.7
        )
    plt.legend(loc="lower right")
    plt.text(
            1.05, 1.05, corr_str,
            transform=plt.gca().transAxes, fontsize=15, verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
    )
    # Update the x-tick labels to include sample counts
    group_counts = extended_df.groupby(group_by_col).size() / 2
    # Add correlation text to the top of each violin
    xticks = ax.get_xticks()
    xtick_labels = []
    for i, group in enumerate(extended_df[group_by_col].unique()):
        rho, p = spearman_results[group]
        if rho is not None:  # If correlation was computed
            text = f"ρs.:{rho:.2f}\np:{p:.2g}"
        else:
            text = ""
        ax.text(
                xticks[i],  # Position at the x-tick of the group
                ax.get_ylim()[1] - 0.051 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                # Just above the violin
                text,
                ha='center', fontsize=6, color='black'
        )
        xtick_labels.append(f"{group}\n(n={group_counts[group]})")
    ax.set_xticklabels(xtick_labels)
    # Set plot title and labels
    if plot_title is None:
        plot_title = f'{prefix.capitalize()} Comparison of {metric1_label} and {metric2_label} Grouped by {group_by_col}'
    plt.title("\n".join(wrap(plot_title, 60)))
    plt.xlabel(group_by_col)
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path.join(save_dir, f"{prefix}-violin-grouped-by-{group_by_col}.png"))
    plt.close()

def train_and_evaluate_model(model_name, X_train, X_test, y_train, y_test, num_epochs, regress_or_classify="regress"):
    """
    Trains and evaluates a specified model on the given dataset.

    Parameters:
    - model_name: Name of the model (e.g., 'DNN', 'SVM', 'RF', 'XGBoost').
    - X_train: Training features.
    - X_test: Testing features.
    - y_train: Training labels. array
    - y_test: Testing labels.
    - num_epochs: Number of epochs (for DNN).
    - output_units: Number of output units (for DNN).

    Returns:
    - history: Training history (if applicable).
    - model: Trained model.
    """
    if "regre" in regress_or_classify:
        model = get_regression_model_given_name(model_name, input_shape=X_train.shape[1])
    elif "classi" in regress_or_classify:
        model = get_classifier_model_given_name(model_name, input_shape=X_train.shape[1])

    if model_name == 'DNN' or model_name.lower() == "fnn":
        history, model = build_and_train_models(model,
            X_train, y_train, X_test, y_test, num_epochs=num_epochs)
    else:
        model.fit(X_train, y_train)
        history = None

    return history, model

def prepare_train_test_data(X_train, y_train, tcga_X, tcga_y, train_set_name="GDSC",
                            test_set_name="TCGA"):
    if train_set_name == "GDSC" and test_set_name == "TCGA":

        used_tcga_X_test = tcga_X.copy()
        used_tcga_y_test = tcga_y.copy()
    elif train_set_name == "GDSC+TCGA" and test_set_name == "TCGA":
        train_tcga_y, val_tcga_y = train_test_split(tcga_y, test_size=0.2, random_state=89)
        train_tcga_x = tcga_X[tcga_y["short_sample_id"].isin(train_tcga_y["short_sample_id"])]
        val_tcga_x = tcga_X[tcga_y["short_sample_id"].isin(val_tcga_y["short_sample_id"])]

        X_train = pd.concat([X_train, train_tcga_x], axis=0)
        y_train = pd.concat([y_train, train_tcga_y], axis=0)

        used_tcga_X_test = val_tcga_x
        used_tcga_y_test = val_tcga_y
    elif train_set_name == "TCGA" and test_set_name == "TCGA":
        train_tcga_y, val_tcga_y = train_test_split(tcga_y, test_size=0.2, random_state=89)
        train_tcga_x = tcga_X[tcga_y["short_sample_id"].isin(train_tcga_y["short_sample_id"])]
        val_tcga_x = tcga_X[tcga_y["short_sample_id"].isin(val_tcga_y["short_sample_id"])]

        X_train = train_tcga_x.copy()  # overwrite train x, y with the TCGA data
        y_train = train_tcga_y.copy()

        used_tcga_X_test = val_tcga_x
        used_tcga_y_test = val_tcga_y

    return X_train, y_train, used_tcga_X_test, used_tcga_y_test


def combine_dicts(dict1, dict2, merge_func=lambda x, y: x + y):
    """
    Combine two dictionaries with the same keys by merging their values using a custom function.

    Args:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.
        merge_func (function): A function that defines how to merge values for the same key.

    Returns:
        dict: A new dictionary with combined values.
    """
    return {key: np.concatenate((dict1[key], dict2[key])) for key in dict1.keys()}

def make_classifiers(tcga_mix: GExMix,
                     train_val_x: pd.DataFrame,
                     train_val_y,
                     overall_results_df: pd.DataFrame,
                     epochs: int,
                     runs: int,
                     model_names: list,
                     tcga_X_test=None,
                     tcga_y_test=None,
                     train_set_name="GDSC",
                     test_set_name="TCGA",
                     y_target_col="binary_response",
                     postfix=".", drug="5fu",
                     regress_or_classify="classification", save_dir="./", aug_by_col="binary_response"):
    """
    Given gene expression data x and meta data y, with label column, do classification
    :param X_train: pd.Dataframe
    :param y_train: dict
    :param tcga_X_test: pd.Dataframe
    :param tcga_y_test: np.array
    :param class_count_str:
    :param label_col:
    :param epochs:
    :param postfix:
    :param regress_or_classify:
    :param save_dir:
    :return:
    """

    # Train the model

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # model_name = "FNN"
    # Resample the training set to handle the imbalance
    # ros = SMOTE(random_state=42)
    # X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    # Initialize an empty list to store results from all folds
    allCV_results = []

    class_count_str = f"{train_set_name}->{test_set_name}"  ## "GDSC+TCGA -> TCGA"
    aug_by_col = "binary_response"
    for model_name in model_names:
        for run in range(runs):
            run_postfix = f"run{run}-{postfix}"
            gdsc_kf = kf.split(train_val_x)
            tcga_kf = kf.split(tcga_X_test)
            for fold_idx, (train_index_gdsc, val_index_gdsc), (train_index_tcga, val_index_tcga) in zip(np.arange(n_splits), gdsc_kf, tcga_kf):

                X_train, X_val = train_val_x.iloc[train_index_gdsc], train_val_x.iloc[val_index_gdsc]
                y_train, y_val = train_val_y.iloc[train_index_gdsc], train_val_y.iloc[val_index_gdsc]

                train_tcga_x = tcga_X_test.iloc[train_index_tcga]
                train_tcga_y = tcga_y_test.iloc[train_index_tcga]
                used_tcga_X_test = tcga_X_test.iloc[val_index_tcga]
                used_tcga_y_test = tcga_y_test.iloc[val_index_tcga]

                if train_set_name == "GDSC+TCGA" and test_set_name == "TCGA":
                    X_train = pd.concat([X_train, train_tcga_x], axis=0)
                    y_train = pd.concat([y_train, train_tcga_y], axis=0)

                elif train_set_name == "TCGA" and test_set_name == "TCGA":
                    X_train = train_tcga_x.copy()  # overwrite train x, y with the TCGA data
                    y_train = train_tcga_y.copy()

                # for model_name in ["FNN", "linearregression"]:
                print(f"{model_name}: {run_postfix} original")

                overall_results_df, gt_pred_df = train_evaluate_and_log(
                    model_name=model_name,
                    X_train=X_train.values,
                    X_val=X_val.values,
                    y_train_df=y_train,
                    y_val_df=y_val,
                    tcga_X_test=used_tcga_X_test.values,
                    tcga_y_test_df=pd.DataFrame(used_tcga_y_test),
                    num_epochs=epochs,
                    run=run,
                    drug=drug,
                    ori_or_aug="ori",
                    y_target_col=y_target_col,
                    aug_by_col="",
                    save_dir=save_dir,
                    run_postfix=run_postfix+f"-ori-run{run}-cv{fold_idx}-{model_name}",
                    overall_results_df=overall_results_df,
                    title_suffix=f"\n{class_count_str}",
                    regress_or_classify=regress_or_classify
                )

                # Create a DataFrame to store fold results
                temp_results = pd.DataFrame(
                        {
                                "Drug": [drug] * len(gt_pred_df),
                                "ori_or_aug": ["ori"] * len(gt_pred_df),
                                "Run": [run] * len(gt_pred_df),
                                "Fold": [fold_idx] * len(gt_pred_df),
                                "model_name": [model_name] * len(gt_pred_df)
                        })

                # Concatenate with additional meta data
                fold_drug_results = pd.concat(
                        [temp_results, gt_pred_df, used_tcga_y_test.reset_index(drop=True)], axis=1)

                # Append fold results to the list
                allCV_results.append(fold_drug_results)

                if tcga_mix.num2aug > 0:
                    aug_gex3, aug_label_dict3 = tcga_mix.augment_adding_gaussian_noise(
                        aug_by_col, target_features=X_train.values,
                        target_label_dict=y_train.reset_index(drop=True),
                        keys4mix=["binary_response", "diagnosis", "dataset_name"],
                            keys4mix_onehot=[],  # doing classification, then use "classification"
                        if_include_original=True, save_dir=save_dir
                    )
                    visualize_data_with_meta(
                            pd.DataFrame(aug_gex3),
                            pd.DataFrame(aug_label_dict3),
                            ["binary_response", "diagnosis", "dataset_name"],
                            postfix=f"{drug}-Aug-{model_name}-run{run}-cv{fold_idx}",
                            if_color_gradient=[True, False, False],
                            cmap="viridis", figsize=[8, 8], vis_mode="umap",
                            save_dir=save_dir
                    )
                else:
                    aug_gex3 = X_train
                    aug_label_dict3 = y_train

                # aug_gex, aug_label_dict = tcga_mix.augment_random(
                #     aug_by_col, target_features=X_train.values,
                #     target_label_dict=y_train.reset_index(drop=True),
                #     keys4mix=["binary_response", "diagnosis", "dataset_name"],
                #         keys4mix_onehot=[],  # doing classification, then use "classification"
                #     if_include_original=True, save_dir=save_dir
                # )
                # aug_gex2, aug_label_dict2 = tcga_mix.augment_random(
                #     "dataset_name", target_features=X_train.values,
                #     target_label_dict=y_train.reset_index(drop=True), num2aug=50,
                #     keys4mix=["binary_response", "diagnosis", "dataset_name"],
                #         keys4mix_onehot=[],  # doing classification, then use "classification"
                #     if_include_original=False, save_dir=save_dir
                # )
                # aug_gex3 = pd.concat([pd.DataFrame(aug_gex), pd.DataFrame(aug_gex2)], axis=0)
                # # Combine by creating lists of values
                # aug_label_dict3 = combine_dicts(aug_label_dict, aug_label_dict2, merge_func=lambda x, y: [x, y])
                #
                visualize_data_with_meta(
                        pd.DataFrame(X_train),
                        pd.DataFrame(y_train),
                        ["binary_response", "diagnosis", "dataset_name"],
                        postfix=f"{drug}-Ori-{model_name}-run{run}-cv{fold_idx}",
                        if_color_gradient=[True, False, False],
                        cmap="viridis", figsize=[8, 8], vis_mode="umap",
                        save_dir=save_dir
                )

                print(f"{model_name}: {run_postfix} augment")
                overall_results_df, gt_pred_df = train_evaluate_and_log(
                    model_name=model_name,
                    X_train=aug_gex3,
                    y_train_df=pd.DataFrame(aug_label_dict3),
                    X_val=X_val.values,
                    y_val_df=y_val,
                    tcga_X_test=used_tcga_X_test.values,
                    tcga_y_test_df=pd.DataFrame(used_tcga_y_test),
                    num_epochs=epochs,
                    run=run,
                    drug=drug,
                    ori_or_aug="aug",
                    y_target_col=y_target_col,
                    save_dir=save_dir,
                    title_suffix=f"\n{class_count_str} ",
                    run_postfix=run_postfix+f"-augby-{aug_by_col[0:5]}{tcga_mix.num2aug}-run{run}-cv{fold_idx}-{model_name}",
                    overall_results_df=overall_results_df,
                    regress_or_classify=regress_or_classify
                )

                # Create a DataFrame to store fold results
                temp_results = pd.DataFrame(
                        {
                                "Drug": [drug] * len(gt_pred_df),
                                "ori_or_aug": ["aug"] * len(gt_pred_df),
                                "Run":  [run] * len(gt_pred_df),
                                "Fold": [fold_idx] * len(gt_pred_df),
                                "model_name": [model_name] * len(gt_pred_df)
                        })

                # Concatenate with additional meta data
                fold_drug_results = pd.concat(
                        [temp_results, gt_pred_df, used_tcga_y_test.reset_index(drop=True)], axis=1)

                # Append fold results to the list
                allCV_results.append(fold_drug_results)
    # Combine results from all folds into a single DataFrame
    allCV_results_df = pd.concat(allCV_results, ignore_index=True)

    return overall_results_df, allCV_results_df


def check_GDSC_TCGA_col_stats( y_df, col2check, group_by_col='TCGA Classification', separate_by="dataset_name",
                     prefix="prefix", save_dir="./"):
    unique_datasets = y_df[separate_by].unique()
    fig, axes = plt.subplots(
            len(unique_datasets), 1,
            figsize=(10, 6 + 2 * len(unique_datasets)),
            sharex=False, sharey=False)
    if len(unique_datasets) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for axe, dataset_name in zip(axes, unique_datasets):
        # Filter data for the current dataset
        dataset_df = y_df[y_df[separate_by] == dataset_name]

        data_counts = pd.crosstab(dataset_df["diagnosis"], dataset_df[col2check])
        # Stacked bars
        categories = data_counts.columns  # The discrete values (e.g., 0, 0.5, 1)
        bottom_value = None  # To stack the bars
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Define colors for each category (optional)
        for i, category in enumerate(categories):
            axe.bar(
                    data_counts.index,  # Group labels
                    data_counts[category],  # Height of the bars (counts for this category)
                    bottom=bottom_value,  # Stack on top of the previous category
                    label=f"{col2check} {category}",
                    color=colors[i] if i < len(colors) else None  # Use specified colors or default
            )
            bottom_value = data_counts[category] if bottom_value is None else bottom_value + \
                                                                              data_counts[category]
            axe.legend(title="Diagnosis")
            axe.tick_params(axis='x', rotation=45)
            axe.set_ylabel("Number of Samples")
            # Add labels and formatting
            axe.set_title(f"{dataset_name} response stats")
    axe.set_xlabel("Group (Count)")

    plt.xticks(ha='right')
    plt.tight_layout()
    # Save and close plot
    plt.savefig(path.join(save_dir, f"{prefix}-stats-{group_by_col}.png"))
    plt.close()

def plot_prediction_gt_scatter(y_pred, y_test_df, title="title", y_target_col="y_target_col",
                               postfix="postfix", save_dir="./"):
    if len(y_pred.shape) == 2:
        y_pred_new = y_pred[:, 0]
    elif len(y_pred.shape) == 1:
        y_pred_new = y_pred

    y_test_values = y_test_df[y_target_col].values

    fig, axes = plt.subplots(1, 2, figsize=[13, 6])
    plt.suptitle(title, fontsize=16)
    # Plot scatter plot of predictions vs. ground truth
    axes[0].scatter(y_test_values + np.random.uniform(0, 0.05, len(y_test_df)), y_pred_new, alpha=0.7)
    axes[0].set_title("Prediction vs. Ground Truth")
    axes[0].set_xlabel("Ground Truth")
    axes[0].set_ylabel("Prediction")
    axes[0].plot([0, 1], [0, 1], color='red', linestyle='--')
    # Plot scatter plot of predictions vs. sample index
    axes[1].scatter(np.arange(len(y_pred)), y_pred, marker="d", alpha=0.7, label="Pred.")
    axes[1].scatter(np.arange(len(y_test_df)), y_test_values, marker="*", alpha=0.7, label="GT")

    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel(y_target_col)
    axes[1].legend()
    # Draw lines connecting corresponding pairs

    for i in range(len(y_test_values)):
        axes[1].plot([i, i], [y_test_values[i], y_pred_new[i]], color='grey', linestyle='--')
    plt.tight_layout()

    plt.savefig(path.join(save_dir, f"scatter_pred_gt_{postfix}.png"))
    plt.close()


def evaluate_model(model, X_test, y_test, epsilon=0.2):
    y_pred = model.predict(X_test)
    target_values = np.array([0, 0.5, 1])
    rounded_predictions = round_to_closest_target(y_pred, target_values)
    # Calculate accuracy based on the tolerance threshold
    accuracy = np.mean(np.abs(rounded_predictions - y_test) <= epsilon)

    return y_pred, accuracy


def plot_history_train_val(history, title="title", postfix="run_postfix", save_dir="save_dir"):
    # Plot each key in a separate subplot
    keys = list(history.history.keys())
    num_metrics = len(keys) // 2
    fig, axs = plt.subplots(num_metrics, 1, figsize=(10, num_metrics * 4))
    if num_metrics == 1:
        axs = [axs]
    try:
        plt.suptitle(title)
        for i, key in enumerate([k for k in keys if 'val_' not in k]):
            axs[i].plot(history.history[key], label=f'Training {key}')
            axs[i].plot(history.history[f'val_{key}'], label=f'Validation {key}')
            axs[i].set_title(f'{key} over Epochs')
            axs[i].set_ylabel(key)
            axs[i].set_xlabel('Epoch')
            axs[i].legend(loc='upper left')
            axs[i].grid(True)
        plt.tight_layout()
        plt.savefig(path.join(save_dir, f"learning_curve_{postfix}.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting history: {e}")


def process_data_handle_duplicates(x, y):
    duplicated = y["bcr_drug_barcode"].duplicated()

    drop_dup_x = x[~duplicated]
    drop_dup_y = y[~duplicated]
    #
    drop_dup_nonan_y = drop_dup_y[drop_dup_y['binary_response'].notnull()]
    drop_dup_nonan_x = drop_dup_x[drop_dup_y['binary_response'].notnull()]

    return drop_dup_nonan_x, drop_dup_nonan_y


# Function to clean up the 'consensus_response' column
def clean_consensus_response(response):
    if pd.isna(response):  # Handle NaN values
        return response
    return response.replace("NA", "").replace("-///-", "").strip()


def get_train_val_data(drop_dup_nonan_x, drop_dup_nonan_y):
    train_inds, test_inds, y_train, y_test = train_test_split(np.arange(len(drop_dup_nonan_x)),
                                                              drop_dup_nonan_y[
                                                                  "binary_response"].values,
                                                              test_size=0.2,
                                                              random_state=42)
    # Calculate unique values and their counts
    unique_counts = drop_dup_nonan_y["binary_response"].value_counts().sort_index()
    # Format the output as "unique_0(count0)-unique_1(count1)..."
    class_counts_str = '-'.join(
        [f"{index}({count})" for index, count in unique_counts.items()])
    X_train = drop_dup_nonan_x.iloc[train_inds]
    X_test = drop_dup_nonan_x.iloc[test_inds]
    df_y_train = drop_dup_nonan_y.iloc[train_inds]
    # df_y_train.reset_index(drop=True, inplace=True)
    df_y_train_dict = {col: df_y_train[col].to_numpy() for col in
                       df_y_train.columns}
    df_y_test = drop_dup_nonan_y.iloc[test_inds]
    df_y_test_dict = {col: df_y_test[col].to_numpy() for col in
                      df_y_test.columns}
    uniq_y, counts = np.unique(y_train, return_counts=True)

    return X_train, pd.DataFrame(df_y_train_dict), class_counts_str, X_test, pd.DataFrame(df_y_test_dict)


def build_and_train_models(model, X_train, y_train, X_test, y_test, num_epochs):
    """

    :param X_test: np.array
    :param X_train: array or df
    :param y_train: array
    :param num_epochs:
    :param output_units:
    :param y_test:
    :return:
    """
    # EarlyStopping Callback
    early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, y_train, epochs=num_epochs,
                        batch_size=32, validation_data=[X_test, y_test],
                        verbose=1, callbacks=early_stopping)
    return history, model


def DGE_analysis(features_list, meta_list, postfix="postfix", p_val_col2use="correct", save_dir="./"):
    """
    differential gene expression analysis
    1. return DGEA results
    2. plot volcano plot in png
    3. plot interactive volcano plot with bokeh
    :return:
    """

    results = main_DGEA(features_list, meta_list, p_val_col2use, postfix=postfix, save_dir=save_dir)

    # plot volcano plot
    plot_volcano_with_anno(results, postfix="postfix", topK=20, save_dir=save_dir)

    # plot bokeh volcano plots for further gene set enrichment analysis
    hover_notion = []
    s2_hoverkeys = ["-log10_p_value", "log2_fold_change", "gene"]
    for key in s2_hoverkeys:
        hover_notion.append((key, results[key]))
    color_by = "-log10_p_value"
    interactive_bokeh_with_select(results["log2_fold_change"].values,
                                  results["-log10_p_value"].values,
                                  hover_notions=hover_notion,
                                  table_columns=["x", "y"] + s2_hoverkeys,
                                  if_color_gradient=True, height=500, width=500,
                                  color_by="-log10_p_value",
                                  s2_hoverkeys=s2_hoverkeys,
                                  title=f"Test color by {color_by}-{postfix}", mode="umap",
                                  postfix=f"{color_by}-{postfix}",
                                  save_dir=save_dir, scatter_size=6)
    return results


def main_DGEA(features_list, meta_list, p_val_col2use, postfix="postfix", save_dir="./"):
    """
    main differential gene expression analysis
    :param features_list:
    :param meta_list:
    :param p_val_col2use:
    :param postfix:
    :param save_dir:
    :return:
    """
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests
    group1_data, group2_data = features_list[0], features_list[1]
    group1_meta, group2_meta = meta_list[0], meta_list[1]
    # Perform differential expression analysis
    p_values = []
    logfold_change = []
    for ind, gene in enumerate(group2_data.columns):
        group1_values = group1_data.values[:, ind]
        group2_values = group2_data.values[:, ind]
        t_stat, p_val = ttest_ind(group1_values, group2_values)
        fold_change = np.log2(np.mean(group1_values) / np.mean(group2_values))
        p_values.append(p_val)
        logfold_change.append(fold_change)
    # logfold_change = zscore(logfold_change, axis=1)
    # Step 1: Identify and filter out NaN values
    valid_p_values = np.array(p_values)[np.isfinite(p_values) & ~np.isnan(p_values)]
    # Step 2: Apply the Benjamini-Hochberg correction
    corrected_valid_p_values = multipletests(valid_p_values, method='fdr_bh')[1]
    # Step 3: Reconstruct the corrected p-values array, maintaining NaN positions
    p_values_corrected = np.full_like(p_values, np.nan)
    p_values_corrected[np.isfinite(p_values) & ~np.isnan(p_values)] = corrected_valid_p_values
    # Create a DataFrame to hold results
    results = pd.DataFrame({
        'gene': group2_data.columns,
        'p_value': p_values,
        'p_value_corrected': p_values_corrected,
        "log2_fold_change": logfold_change
    })
    # Add a column for significance
    if "correct" in p_val_col2use:
        col = 'p_value_corrected'
    else:
        col = 'p_value'
    results['-log10_p_value'] = -np.log10(results[col])
    log2_fc_cutoff = np.percentile(
        abs(results[results['log2_fold_change'].notna()]['log2_fold_change']), 80)
    # Classify the data points based on the cut-offs
    results['significant'] = (
                results[col] < 0.05)  # (abs(results['log2_fold_change']) >= log2_fc_cutoff) &
    results.sort_values(["-log10_p_value", 'log2_fold_change'], inplace=True, ascending=False)
    # Save results to a CSV file
    results.to_csv(path.join(save_dir, f'DGEA_{postfix}.csv'), index=False)

    return results


def plot_volcano_with_anno(results, postfix="postfix", topK=10, save_dir="./"):
    """
    results from differential gene expression analysis
    :param results:
    :param postfix:
    :param save_dir:
    :return:
    """
    from adjustText import adjust_text
    # Create the volcano plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=results, x='log2_fold_change', y='-log10_p_value', hue='significant',
                    palette={True: 'tab:green', False: 'grey'}, legend=False)
    # Highlight significant genes
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 P-value')
    plt.title(f'DGE {postfix} top{topK}')
    plt.grid(True)

    # get topK genes to annotate
    neg_top_sig_data = results.sort_values(["p_value_corrected"],
                                           ascending=True).iloc[0:topK]
    neg_top_sig_data = results.sort_values(["log2_fold_change"],
                                           ascending=True).iloc[0:topK]
    pos_top_sig_data = results.sort_values(["log2_fold_change", "p_value_corrected"],
                                           ascending=False).iloc[0:topK]

    # Annotate top genes for high p-values
    texts = []
    for i in range(topK):
        texts.append(plt.text(neg_top_sig_data["log2_fold_change"].values[i],
                              neg_top_sig_data["-log10_p_value"].values[i],
                              neg_top_sig_data["gene"].values[i], fontsize=9))
    # Annotate top genes for high log2fold values
    for i in range(topK):
        texts.append(plt.text(pos_top_sig_data["log2_fold_change"].values[i],
                              pos_top_sig_data["-log10_p_value"].values[i],
                              pos_top_sig_data["gene"].values[i], fontsize=9))
    # Adjust text to minimize overlap
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))
    plt.savefig(path.join(save_dir, f'DGEA_{postfix}_volcano_anno.png'))
    plt.close()


def gene_set_enrichment_analysis(gene_list, result_file, topK=10, key2use="Adjusted P-value", postfix="stad", save_dir="./"):
    import gseapy as gp
    enr = gp.enrichr(gene_list=gene_list,  # or "./tests/data/gene_list.txt",
                     gene_sets=['MSigDB_Hallmark_2020', 'Human_Phenotype_Ontology', "WikiPathway_2023_Human", "Reactome_2022"], #
                     organism='human',
                     # don't forget to set organism to the one you desired! e.g. Yeast
                     outdir=None,  # don't write to disk
                     )
    geneset_color = {'KEGG_2021_Human': 'tab:orange',
                     'MSigDB_Hallmark_2020': 'tab:blue',
                     'WikiPathway_2023_Human': 'tab:green',
                     'Human_Phenotype_Ontology': 'tab:purple',

                     }
    # categorical scatterplot
    ax = gp.dotplot(enr.results,
                 column="Adjusted P-value",
                 x='Gene_set',  # set x axis, so you could do a multi-sample/library comparsion
                 size=10,
                 top_term=5,
                 figsize=(6, 9),
                 title=postfix,
                 xticklabels_rot=45,  # rotate xtick labels
                 show_ring=True,  # set to False to revmove outer ring
                 marker='o',
                    ofname=path.join(save_dir,
                                     f"enrichr_dot_{postfix}.png")
                 )
    ax = gp.barplot(enr.results,
                    column="Adjusted P-value",
                    group='Gene_set',
                    # set group, so you could do a multi-sample/library comparsion
                    size=10,
                    top_term=5,
                    title=postfix,
                    figsize=(6, 9),
                    # color=['darkred', 'darkblue'] # set colors for group
                    color=geneset_color,
                    ofname=path.join(save_dir,
                                     f"enrichr_bar_{key2use}_{postfix}.png"))
    # Perform GSEA
    pre_res = gp.prerank(rnk=result_file,   # result file from differential gene expressio nanalysis
                         gene_sets='WikiPathway_2023_Human',
                         threads=4,
                         min_size=5,
                         max_size=1000,
                         permutation_num=1000,  # reduce number to speed up testing
                         outdir=None,  # don't write to disk
                         seed=6,
                         verbose=True)  # see what's going on behind the scenes
    # Extract significant results
    res_df = pre_res.res2d
    significant_res = res_df[res_df['NOM p-val'] < 0.05]
    # Split positive and negative enrichment scores
    pos_enrichment = pd.DataFrame(significant_res[significant_res['NES'] > 0])
    neg_enrichment = pd.DataFrame(significant_res[significant_res['NES'] < 0])
    pos_enrichment["NES"] = pos_enrichment["NES"].astype(np.float32)
    neg_enrichment["NES"] = neg_enrichment["NES"].astype(np.float32)

    # Sort by normalized enrichment score (NES)
    pos_enrichment = pos_enrichment.sort_values(by='NES', ascending=True)
    neg_enrichment = neg_enrichment.sort_values(by='NES', ascending=True)

    # Select the top K pathways from positive and negative enrichments
    top_pos_enrichment = pos_enrichment.nlargest(min(topK, len(pos_enrichment)), 'NES', keep='all')
    top_neg_enrichment = neg_enrichment.nsmallest(min(topK, len(neg_enrichment)), 'NES', keep='all')

    # Combine positive and negative enrichment scores
    combined_scores = np.concatenate([top_pos_enrichment['NES'].values, top_neg_enrichment['NES'].values])
    combined_labels = np.concatenate([top_pos_enrichment['Term'], top_neg_enrichment['Term']])

    # Create two-sided barplot
    fig, ax = plt.subplots(figsize=(10, 8))
    # Plot positive enrichment (right side)
    ax.barh(np.arange(len(top_pos_enrichment)), top_pos_enrichment['NES'], color='tab:blue',
            edgecolor='black', label='Positive NES')
    # Plot negative enrichment (left side)
    ax.barh(np.arange(len(top_pos_enrichment), len(combined_scores)), top_neg_enrichment['NES'],
            color='tab:orange', edgecolor='black', label='Negative NES')
    # Set labels and title
    ax.set_xlabel('Normalized Enrichment Score (NES)')
    ax.set_title(f'GSEA Results:{differential_postfix}')
    ax.legend()
    # Customize the y-ticks to show both positive and negative indices
    ax.set_yticks(np.arange(len(combined_scores)))
    ax.set_yticklabels(combined_labels)
    ax.axvline(x=0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(path.join(save_dir, f"DGEA_two-side-barplot-{postfix}.png"))
    return enr.results

def select_top_k_percent_std(df, num_columns_to_select):
    """
    Select the top K percent of columns with the highest standard deviation.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    k (float): The percentage (between 0 and 100) of columns to select.

    Returns:
    pd.DataFrame: A DataFrame containing only the selected columns.
    """
    # Calculate the standard deviation for each column
    std_devs = df.std()

    # Determine the number of columns to select
    # num_columns_to_select = int(len(std_devs) * (k / 100))
    num_columns_to_select = int(num_columns_to_select)

    # Select the top K percent columns based on standard deviation
    top_k_columns = std_devs.nlargest(num_columns_to_select).index

    # Return the DataFrame with only the selected columns
    return df[top_k_columns]


## 2024.11.20

def prepare_data_for_regressor(gdsc_df, gex_all, meta_all):
    """
    For each drug in gdsc_df, prepare a dataset of gene expression data (from gex_all)
    and drug response (from gdsc_df) for training a regressor.

    Args:
        gdsc_df (pd.DataFrame): DataFrame containing cell line-drug pairs and drug response.
        gex_all (pd.DataFrame): DataFrame containing gene expression data for cell lines.
        meta_all (pd.DataFrame): DataFrame containing metadata of cell lines (optional in this context).

    Returns:
        dict: A dictionary where each drug is a key, and the value is a tuple (X, y),
              with X being the gene expression data and y being the drug response.
    """
    # Ensure all cell line names are in the same case for consistency
    gdsc_df["cell_line_name"] = gdsc_df["cell_line_name"].str.upper()
    gex_all["cell_line_name"] = gex_all["cell_line_name"].str.upper()
    meta_all["cell_line_name"] = meta_all["cell_line_name"].str.upper()

    # Initialize a dictionary to store the training data for each drug
    drug_data_dict = {}

    # Iterate over each unique drug in gdsc_df
    for drug in gdsc_df["drug_name"].unique():
        # Filter gdsc_df for the current drug
        drug_df = gdsc_df[gdsc_df["drug_name"] == drug]

        # Find the cell lines that are common between gdsc_df (for this drug) and gex_all
        common_cell_lines = set(drug_df["cell_line_name"]).intersection(set(gex_all["cell_line_name"]))

        # Filter both gdsc_df and gex_all to keep only the common cell lines
        filtered_gdsc = drug_df[drug_df["cell_line_name"].isin(common_cell_lines)]
        filtered_gex = gex_all[gex_all["cell_line_name"].isin(common_cell_lines)]

        # Align the dataframes by cell line name
        filtered_gdsc = filtered_gdsc.set_index("cell_line_name")
        filtered_gex = filtered_gex.set_index("cell_line_name")

        # Ensure the cell lines are aligned
        filtered_gex = filtered_gex.loc[filtered_gdsc.index]

        # Prepare the input (gene expression) and output (drug response)
        X = filtered_gex.drop(columns=["cell_line_name"], errors="ignore").values
        y = filtered_gdsc["drug_response"].values  # Replace "drug_response" with the actual column name

        # Store the data in the dictionary
        drug_data_dict[drug] = (X, y)

    return drug_data_dict

def process_GDSC_raw_gene_expression():

    def standardize_cell_line_name(cell_line_name, exceptions):
        """
        Standardize cell line names by stripping hyphens in GDSC to match the cell line names in CCLE, where - are stripped\
        , except for those in the exceptions list.

        Args:
            cell_line_name (str): The cell line name to standardize.
            exceptions (list): List of cell line names to exclude from hyphen removal.

        Returns:
            str: Standardized cell line name.
        """
        if cell_line_name in exceptions:
            return cell_line_name  # Preserve exceptions as-is
        return cell_line_name.replace(
                "-", "").upper()  # Remove hyphens and standardize to uppercase

    def get_exceptions(dataframe, cell_line_name_col='Cell Line Name', cosmic_id_col="COSMIC_ID"):
        """
        Identify cell line names that, when standardized by removing hyphens,
        map to the same name but have different COSMIC IDs, indicating they are distinct cell lines.

        Args:
            dataframe (pd.DataFrame): The input dataframe containing 'cell_line_name' and 'cosmic_id'.

        Returns:
            set: A set of cell line names that should be treated as exceptions during standardization.
        """
        df = dataframe.copy()
        # Standardize cell line names by removing hyphens and converting to uppercase
        df['standardized_name'] = df[cell_line_name_col].str.replace('-', '').str.upper()
        # Group by the standardized names
        grouped = df.groupby('standardized_name')
        # Initialize a set to hold exceptions
        exception_list = []
        # Iterate over the groups
        for name, group in grouped:
            # If the group contains more than one unique original cell line name with different COSMIC IDs
            if group[cell_line_name_col].nunique() > 1 and group[cosmic_id_col].nunique() > 1:
                # Add all original cell line names in this group to the exceptions set
                exception_list += list(group[cell_line_name_col].unique())
        print(f"Exception name list: {exception_list}")
        return exception_list

    """
    From online downloaded files, with gene expression has array names
    1. extract cell line name of the array names from E-MTAB-3610E-MTAB-3610.sdrf.txt
    2. realign meta data with gene expression data
    3. deal with cell line w/wo "-", especially cases like T-T, and TT
    4. add relevant columns for later merging with CCLE
    :return:
    """
    # File paths
    cell_array_names_path = r"..\data\GDSC_drug_response\GDSC-cellline-meta-E-MTAB-3610.sdrf.txt"
    gdsc_expression_path =  r"..\data\GDSC_drug_response\GDSC_gene_expression_data_GeneSymbols (1).csv"
    # Read data
    cell_array_meta = pd.read_csv(cell_array_names_path, sep="\t")
    gdsc_expression = pd.read_csv(gdsc_expression_path)
    # Extract array names and map to cell line names
    array_names = list(gdsc_expression.columns)[1:]  # Exclude the first column (GeneSymbol)
    cell_name_map = dict(zip(cell_array_meta["Array Data File"], cell_array_meta["Source Name"]))
    # Map or retain original array name if no match found
    extracted_cell_names = [cell_name_map.get(arr, arr) for arr in array_names]
    # # Rename columns
    gdsc_expression.columns = ["GeneSymbol"] + extracted_cell_names
    # Check the updated DataFrame
    # print(gdsc_expression.head())
    transposed_df = gdsc_expression.set_index('GeneSymbol').transpose()

    transposed_df.index = ["_".join(ele.split("_")[:3]) for ele in transposed_df.index]
    # Extract the new cell line order from transposed_df.index
    cell_line_order = transposed_df.index.tolist()
    # # Reorder cell_names dataframe based on the new order
    cell_array_meta.index = ["_".join(ele.split("_")[:3]) for ele in cell_array_meta[
        "Source Name"]]  # Replace 'CellLineName' with the actual column name
    cell_array_meta = cell_array_meta.loc[
        cell_line_order]  # Reorder based on the new cell line order

    cellline_annotation_filename = (r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active "
                                    r"projects\1-pancan_subtyping\data\GDSC_drug_response\Cell_Lines_Details.xlsx")
    cell_line_meta = pd.read_excel(cellline_annotation_filename)

    ## get cell line clinical matching meta data. some cell lines are missing
    # Perform the merge
    merged_df = cell_line_meta.merge(
            cell_array_meta,
            left_on="Sample Name",
            right_on="Characteristics[cell line]",
            how="inner"  # Use 'inner', 'left', or 'right' depending on the desired merge behavior
    )
    merged_df.index = ["_".join(ele.split("_")[:3]) for ele in merged_df["Source Name"]]
    merged_df.insert(3, "diagnosis", merged_df["Cancer Type\n(matching TCGA label)"].values)

    merged_gene_expression = transposed_df.loc[merged_df.index]

    transposed_df.to_csv(
            path.join(
                    "../data/GDSC_drug_response",
                    f"GDSC_gene_expression_{transposed_df.shape}_processed9.csv"), index=False)
    cell_array_meta.to_csv(
            path.join(
                    "../data/GDSC_drug_response",
                    f"GDSC_meta_data_{cell_array_meta.shape}_processed9.csv"), index=False)
    merged_df.to_csv(
            path.join(
                    "../data/GDSC_drug_response",
                    f"GDSC_meta_data_filtered{merged_df.shape}_processed9.csv"), index=False)

    ## to match GexDataset
    merged_df.columns = merged_df.columns.str.replace('\n', '', regex=False)
    merged_df["primary_site"] = merged_df["GDSCTissuedescriptor 2"]
    merged_df["diagnosis_b4_impute"] = merged_df["Cancer Type(matching TCGA label)"]
    merged_df["tumor_percent"] = 100
    merged_df["normal_percent"] = 0
    merged_df["COSMIC_ID"] = merged_df["COSMIC identifier"]
    merged_df["source_labels"] = "gdsc"
    merged_df["tumor_stage"] = ''
    merged_df["sample_id"] = merged_df["COSMIC identifier"]
    exception_cell_name_list = get_exceptions(
        merged_df, cell_line_name_col="Sample Name", cosmic_id_col="COSMIC identifier")
    merged_df["stripped_cell_line_name"] = merged_df[
        "Factor Value[cell_line]"].apply(
            lambda x: standardize_cell_line_name(x, exception_cell_name_list)
    )

    with open(
            path.join(
                    "../data/GDSC_drug_response",
                    f"GDSC_gex_w_filtered_meta_{merged_gene_expression.shape}_processed9.pickle"
            ), 'wb'
    ) as handle:
        gdsc_data_dict = {}
        gdsc_data_dict["rnaseq"] = merged_gene_expression
        gdsc_data_dict["meta"] = merged_df
        pickle.dump(gdsc_data_dict, handle)

    # with open(path.join(
    #                     "../data/GDSC_drug_response",
    #                     f"GDSC_gex_w_filtered_meta_{merged_gene_expression.shape}_processed9.pickle"
    #             , 'rb') as handle:
    #     gdsc_data_dict = pickle.load(handle)
    #     gex_combat_data3 = gdsc_data_dict["rnaseq"]
    #     meta_combat_data3 = gdsc_data_dict["meta"]




def clinical_tcga_investigation(dataset, gexmix, tcga_clinical_filename, save_dir="./"):
    """
    analysis tcga data with availabel clinical data
    1. filter availabel tcga_meta and tcga_gex, with their clinical data
    2. analyze per drug
        2.1 get drug specific availabel tcga_meta and tcga_gex, with their clinical data
        2.2 visualize clinical data
        2.3 augment these data with random balanced method
        2.4 visualize again with clinical data
        2.5 in bokeh, select interested clusters
    :param dataset:
    :param gexmix:
    :param tcga_clinical_filename:
    :param save_dir:
    :return:
    """
    save_dir_base = "../results/GDSC-RES"
    now_time = datetime.now()
    time_str = (f"{now_time.month}-{now_time.day}T{now_time.hour}-"
                f"{now_time.minute}-beta{gexmix.beta_param}-num{gexmix.num2aug}-TCGA-TCGA")
    save_dir = path.join(save_dir_base, time_str)
    if not path.exists(save_dir):
        makedirs(save_dir, exist_ok=True)

    tcga_clinical_dataset = TCGAClinicalDataProcessor(
        dataset, tcga_clinical_filename, save_dir=save_dir)
    gex_clinical_from_tcga, meta_with_clinical_from_tcga, clinical_tcga_valid = (
            tcga_clinical_dataset.get_matched_gex_meta_and_response())
    drug_counts = tcga_clinical_dataset.get_unique_drug_counts()

    num_genes = tcga_clinical_dataset.num_genes
    timestamp = datetime.now().strftime("%m-%dT%H-%M")


    ## visualize original data with different meta with PHATE
    # analyze_original_stats(
    #         clinical_tcga_valid, gex_clinical_from_tcga, gexmix,
    #         meta_with_clinical_from_tcga, num_genes, save_dir=save_dir, timestamp=timestamp
    # )
    # analyze_per_cancer(
    #     clinical_tcga_valid, drug_counts, gex_clinical_from_tcga, gexmix,
    #     meta_with_clinical_from_tcga, save_dir=save_dir, timestamp=timestamp
    #     )

    ## Analyze per drug data, augment, find markers
    """
    1. for each drug used in TCGA, response predicton w/wo DA, could leave out some patients and then rank the drug treatment for them as a validation
    """
    analyze_per_drug_tcga_clinical(
            clinical_tcga_valid, drug_counts, gex_clinical_from_tcga, gexmix,
            meta_with_clinical_from_tcga, num_genes=num_genes, save_dir=save_dir,
            timestamp=timestamp, if_apply_augmentation=True)
    """
    2. work on the combined dataset, train GDSC drug response, w/wo DA, test on CCLE
    See: GDSC_drug_response_with_mix

    3. work on the combined dataset, train GDSC drug response, w/wo DA, test on TCGA: 
        a. predict the response of the tested drugs
        b. propose highly sensitive drugs for tcga samples
    4. need to think about per drug or per cancer?
    5. benchmark representation learning boost with methods from:
        Standard CVAE (Sohn et al., 2015)
        • CVAE with MMD on bottleneck (MMD-CVAE), similar to VFAE (Louizos et al., 2015)
        • MMD-regularized autoencoder (Amodio et al., 2019; Dziugaite
        et al., 2015b)
"""
    print("ok")



def GDSC_DRP_with_mix_with_GDSC(gdsc_gex_dataset, gexmix, save_dir="./"):
    """
    2. work on the combined dataset, train GDSC drug response, w/wo DA, test on CCLE
    """
    """
            3. work on the combined dataset, train GDSC drug response, w/wo DA, test on TCGA: 
                a. predict the response of the tested drugs
                b. propose highly sensitive drugs for tcga samples
            4. need to think about per drug or per cancer?
            5. benchmark representation learning boost with methods from:
                Standard CVAE (Sohn et al., 2015)
                • CVAE with MMD on bottleneck (MMD-CVAE), similar to VFAE (Louizos et al., 2015)
                • MMD-regularized autoencoder (Amodio et al., 2019; Dziugaite
                et al., 2015b)
"""
    gdsc1_dataset = GDSCDataProcessor("GDSC1")
    gdsc2_dataset = GDSCDataProcessor("GDSC2")

    ori_pickle_filename = "file to the drug performance with the original data"

    gex_data_all_filtered = gdsc_gex_dataset.gex_data
    meta_data_all = gdsc_gex_dataset.meta_data
    # top_var_genes = min(2500, gex_data_all.shape[1])
    # gex_data_all_filtered = gdsc1_dataset.get_top_k_most_variant_genes(gex_data_all,
    #                                                                    top_k=top_var_genes)

    split_mode = "leave-cell-line-out"
    num2aug = 400
    betas = [1]
    y_col = "IC50"  ##"AUC"
    model_names_list = ["linearregression", "fnn", "elasticnet"]
    timestamp = datetime.now().strftime("%m-%dT%H-%M")
    aug_by_column = "5IC50"## "TCGA Classification" # "3zscore"  # Catgorize zscore into 6 bins, then augment to cover variance in all bins.

    drug_grouped_data = gdsc1_dataset.load_gdsc_drug_grouped_data(
            gex_data_all_filtered,
            meta_data_all,
            save_prefix="GDSC",
            load_from_saved_filename=r"..\data\GDSC_drug_response\GDSC_403drugs.pickle")  ## "../data/GDSC_drug_response/random_train403_val403_rand42.pickle"

    for jj, beta in enumerate([2]):  # 0.1, 0.5, 1, 2, 4, 5
        # beta = betas[0]
        model_name = model_names_list[0] if len(model_names_list) == 1 else "multiModels"
        model_related_str = f"{model_name}_beta{beta:.1f}" if len(betas) ==1 else "multi-betas"
        # train and save modeled with augmented data
        model_saved_dir = (f"{save_dir}/models/{split_mode}/{timestamp}_{model_related_str}_6z_aug{num2aug}")  ## {aug_by_column[0:5]}

        # Create a model collection and train models
        model_collection = GDSCModelCollection(
            gexmix,
            model_saved_dir)

        copy_save_all_files(model_saved_dir, "./")

        # train and save modeled with original data
        first_k_items = 2

        # train and save modeled with augmented data
        model_collection.performance_metrics = {
                "ori": {
                        "spearman_score": [], "pearson_score": [], "r2_score": [], "pred_w_ori_meta": []
                },
                "aug": {
                        "spearman_score": [], "pearson_score": [], "r2_score": [], "pred_w_ori_meta": []
                }
        }

        ## train with augmentation of training data
        model_collection.train_and_save_models(
                drug_grouped_data, model_names_list, y_col=y_col, gene_start_col=0,
                num2aug=num2aug, use_augmentation=True, betas=[beta],
                keys4mix=["IC50", "AUC", "Z score", "Tissue", "TCGA Classification"],
                aug_by_column=aug_by_column, if_verbose_K=first_k_items
        )

        ## train first with the original data
        if jj == 0:
            model_collection.train_and_save_models(
                    drug_grouped_data, model_names_list,
                    y_col=y_col, gene_start_col=0, use_augmentation=False, if_verbose_K=first_k_items
            )

            saved_filename = path.join(
                    model_saved_dir,
                    f"0-all_models_w_pred.pickle")
            with open(saved_filename, 'wb') as handle:
                pickle.dump(model_collection.performance_metrics, handle)
            ori_pickle_filename = saved_filename

        if path.isfile(ori_pickle_filename):
            with open(ori_pickle_filename, 'rb') as handle:
                model_collection_ori_performance_metrics = pickle.load(handle)
                ori_metrics = model_collection_ori_performance_metrics["ori"]
                model_collection.performance_metrics["ori"] = ori_metrics

            saved_filename = path.join(
                    model_saved_dir,
                    f"0-all_models_w_pred.pickle")
            with open(saved_filename, 'wb') as handle:
                pickle.dump(model_collection.performance_metrics, handle)

        model_collection.plot_overall_performance(model_collection.performance_metrics, prefix=f"0-Aug"
                           f"{num2aug}_beta_{beta:.1f}_{model_name}{model_related_str}", save_dir=model_saved_dir)

    """
        3. work on the combined dataset, train GDSC drug response, w/wo DA, test on TCGA: 
            a. predict the response of the tested drugs
            b. propose highly sensitive drugs for tcga samples
        4. need to think about per drug or per cancer?
        5. benchmark representation learning boost with methods from:
            Standard CVAE (Sohn et al., 2015)
            • CVAE with MMD on bottleneck (MMD-CVAE), similar to VFAE (Louizos et al., 2015)
            • MMD-regularized autoencoder (Amodio et al., 2019; Dziugaite
            et al., 2015b)
            • CycleGAN (Zhu et al., 2017)
        """

def GDSC_DRP_with_mix_with_TCGA(dataset, gexmix, tcga_clinical_filename):
    """
    1. load tcga_drug_response
    2. load GDSC drug response
   3.  get the shared drugs
    4. get train and test data of this drug for GDSC
    5. get train and test data of this drug from TCGA
    6. use categorize_zscore_3_class in GDSCModelCollection to categorize GDSC zscore
    7. train and evaluation with GDSCModelCollection in 5-fold cross validation
        3. work on the combined dataset, train GDSC drug response, w/wo DA, test on TCGA:
            a. predict the response of the tested drugs
            b. propose highly sensitive drugs for tcga samples
        4. need to think about per drug or per cancer?
        5. benchmark representation learning boost with methods from:
            Standard CVAE (Sohn et al., 2015)
            • CVAE with MMD on bottleneck (MMD-CVAE), similar to VFAE (Louizos et al., 2015)
            • MMD-regularized autoencoder (Amodio et al., 2019; Dziugaite
            et al., 2015b)
"""

    # gexmix.beta_param = 0.6
    save_dir_base = "../results/GDSC-RES/GDSC-aug-with-TCGA"
    timestamp = datetime.now().strftime("%m-%dT%H-%M")
    train_set_name = "TCGA"
    test_set_name = "TCGA"
    epochs = 50
    runs = 1
    model_names = ["FNN", "elasticnet", "linearregression"]
    for train_set_name in ["GDSC+TCGA", "GDSC"]:  #"TCGA",  , "GDSC+TCGA"
        for gaussian_scale in [0.1]:  #0.05, 0.1, 0.2 0.2, 0.65,
            gexmix.beta_param = 1
            gexmix.gaussian_scale = gaussian_scale
            for num2aug in [400, 200, 50]:
                gexmix.num2aug = num2aug
                time_str = (f"{timestamp}-{train_set_name}_{test_set_name}_noise{gaussian_scale:.2f}_num{num2aug}")
                save_dir = path.join(save_dir_base, time_str)
                if not path.exists(save_dir):
                    makedirs(save_dir, exist_ok=True)
                copy_save_all_files(save_dir, "./")

                # Step 1: Load TCGA drug response
                tcga_clinical_dataset = TCGAClinicalDataProcessor(
                    dataset, tcga_clinical_filename, save_dir=save_dir)
                gex_clinical_from_tcga, meta_with_clinical_from_tcga, tcga_tumor_drug_pairs = (
                        tcga_clinical_dataset.get_matched_gex_meta_and_response())
                TCGA_drug_counts = tcga_clinical_dataset.get_unique_drug_counts()

                ## Step 2: load GDSC dataset
                gdsc1_dataset = GDSCDataProcessor("GDSC1")

                # Step 3: Get the shared drugs
                shared_drugs = gdsc1_dataset.get_shared_drugs(TCGA_drug_counts, min_num_samples=20)

                num_genes = tcga_clinical_dataset.num_genes

                # Initialize the model collection

                mean_metric_sum_df = pd.DataFrame(
                        columns=["drug", "ori_or_aug", "model_name", "accuracy", "run", "num_samples",
                                 "aug_by_col"])

                for drug_idx, drug_name in enumerate(shared_drugs):
                    if drug_name != "dabrafenib":
                        continue
                    print(f"Processing drug: {drug_name}")
                    drug_save_dir = path.join(save_dir, drug_name)
                    if not path.exists(drug_save_dir):
                        makedirs(drug_save_dir, exist_ok=True)

                    # Step 4: Get data of this drug for GDSC
                    gex_gdsc_x_drug, meta_gdsc_y_drug = gdsc1_dataset.gdsc_extract_data_for_one_drug(
                        drug_name, tcga_clinical_dataset.gex_all, tcga_clinical_dataset.meta_all,
                            drug_identifier_col="Drug Name")
                    meta_gdsc_y_drug["binary_response"] = meta_gdsc_y_drug["Z score"].apply(
                        lambda x: gdsc1_dataset.categorize_zscore_3_class(x))

                    meta_gdsc_y_drug["diagnosis"] = meta_gdsc_y_drug["TCGA Classification"]

                    # Split the dataframe into train and test sets
                    gex_gdsc_x_drug = gex_gdsc_x_drug.iloc[:, -num_genes:]

                    # Step 5: Get train and test data of this drug from TCGA
                    tcga_drug_data = tcga_tumor_drug_pairs[tcga_tumor_drug_pairs['drug_name'] == drug_name]

                    ## get matching gex from gex_clinical_from_tcga
                    gex_tcga_x_drug = gex_clinical_from_tcga[gex_clinical_from_tcga["short_sample_id"].isin(tcga_drug_data["short_sample_id"])].groupby(
                        "short_sample_id").first().reset_index()
                    meta_tcga_y_drug = tcga_drug_data[tcga_drug_data["short_sample_id"].isin(tcga_drug_data["short_sample_id"])].groupby(
                        "short_sample_id").first().reset_index()
                    gex_tcga_x_drug = gex_tcga_x_drug.iloc[:, -num_genes:]

                    meta_gdsc_y_drug["diagnosis"] = meta_gdsc_y_drug["diagnosis"]
                    meta_gdsc_y_drug["dataset_name"] = ["GDSC"] * meta_gdsc_y_drug.shape[0]
                    meta_tcga_y_drug["diagnosis"] = meta_tcga_y_drug["disease_code"]
                    meta_tcga_y_drug["dataset_name"] = ["TCGA"] * meta_tcga_y_drug.shape[0]

                    ## visualize col stats for both GDSC and TCGA
                    visualize_combined_GDSC_TCGA_given_drug(
                            drug_name, [meta_gdsc_y_drug, meta_tcga_y_drug],
                            [gex_gdsc_x_drug, gex_tcga_x_drug],
                            ["GDSC", "TCGA"], col2color=["binary_response", "dataset_name", "diagnosis"],
                            save_dir=drug_save_dir)

                    check_GDSC_TCGA_col_stats(
                            pd.concat([meta_gdsc_y_drug, meta_tcga_y_drug], axis=0), "binary_response",
                            group_by_col=['diagnosis'], save_dir=drug_save_dir, separate_by="dataset_name",
                            prefix=f"{drug_name}")

                    # test the performance for classificaiton + data augmentation
                    mean_metric_sum_df, results_w_prediction_gt_df = make_classifiers(
                            gexmix,
                            train_val_x=gex_gdsc_x_drug,
                            train_val_y=meta_gdsc_y_drug,
                            tcga_X_test=gex_tcga_x_drug,
                            tcga_y_test=meta_tcga_y_drug,
                            test_set_name=test_set_name,  ## TCGA
                            train_set_name=train_set_name,   ## GDSC
                            overall_results_df=mean_metric_sum_df,
                            aug_by_col="binary_response",
                            epochs=epochs,
                            runs=runs,
                            model_names=model_names,
                            y_target_col="binary_response",
                            # aug_by_col and y_target_col don't need to be the same
                            postfix=f"{drug_name}", drug=drug_name,
                            regress_or_classify="regression",
                            save_dir=drug_save_dir
                    )

                    mean_metric_sum_df.to_csv(
                            path.join(
                                    path.dirname(drug_save_dir),
                                    f"{drug_name}-overall_drp_summary.csv"
                            )
                    )
                    results_w_prediction_gt_df.to_csv(
                            path.join(
                                    path.dirname(drug_save_dir),
                                    f"{drug_name}-overall_drp_with_pred_gt_meta.csv"
                            )
                    )
                    print("ok")
    print("Training and evaluation completed.")



    """
        3. work on the combined dataset, train GDSC drug response, w/wo DA, test on TCGA: 
            a. predict the response of the tested drugs
            b. propose highly sensitive drugs for tcga samples
        4. need to think about per drug or per cancer?
        5. benchmark representation learning boost with methods from:
            Standard CVAE (Sohn et al., 2015)
            • CVAE with MMD on bottleneck (MMD-CVAE), similar to VFAE (Louizos et al., 2015)
            • MMD-regularized autoencoder (Amodio et al., 2019; Dziugaite
            et al., 2015b)
            • CycleGAN (Zhu et al., 2017)
        """


def visualize_combined_GDSC_TCGA_given_drug(drug_name, meta_df_list, gex_df_list, dataset_name_list,
                                            col2color=["binary_response", "diagnosis",
                                                       "dataset_name"],
                                            save_dir="./"):
    """
    :param drug_name:
    :param meta_df_list:
    :param gex_df_list:
    :param dataset_name_list:
    :param save_d:
    :return:
    """
    concat_meta = pd.concat(meta_df_list, axis=0)
    concat_features = pd.concat(gex_df_list, axis=0)
    dataset_names = []
    for ind, name in enumerate(dataset_name_list):
        dataset_names += [name] * len(meta_df_list[ind])
    concat_meta["dataset_name"] = dataset_names
    projection = get_reduced_dimension_projection(
        concat_features,
        vis_mode="umap")
    # Ensure NaN values are represented as a distinct category
    concat_meta['binary_response'] = concat_meta['binary_response'].fillna('NaN')
    # Extract unique categories from the binary_response column
    categories = concat_meta['binary_response'].unique()
    # Define a custom palette where 'NaN' is mapped to gray
    base_palette = sns.color_palette("cool", n_colors=len(categories))
    custom_palette = dict(zip(categories[categories != 'NaN'], base_palette))
    custom_palette['NaN'] = (0.8, 0.8, 0.8)  # Explicitly set gray for 'NaN'
    # Scatter plot with hue including NaN
    fig, axes = plt.subplots(1, len(col2color), figsize=(8 + 3 * len(col2color), 7))
    plt.suptitle(f"{drug_name} - GDSC vs TCGA Original Data", fontsize=16)
    for ind, color_by in enumerate(col2color):
        palette = custom_palette if color_by == 'binary_response' else "tab10"
        sns.scatterplot(
                x=projection[:, 0],
                y=projection[:, 1],
                hue=concat_meta[color_by],
                palette=palette,
                alpha=0.7,
                ax=axes[ind]
        )
        axes[ind].set_title(f"colorby {color_by}")

        if color_by == "diagnosis":
            axes[ind].legend(bbox_to_anchor=(1.05, 1), ncols=2)

    plt.tight_layout()
    plt.savefig(path.join(save_dir, f"{drug_name}_GDSC_vs_TCGA_Original_Data.png"))
    plt.close()


def analyze_original_stats(
        clinical_tcga_valid, gex_clinical_from_tcga, gexmix,
        meta_with_clinical_from_tcga, num_genes, save_dir="./", timestamp="."):
    """
    visualize meta data on the original all data
    :param clinical_tcga_valid:
    :param drug_counts:
    :param gex_clinical_from_tcga:
    :param gexmix:
    :param meta_with_clinical_from_tcga:
    :param results_collect_df:
    :param save_dir:
    :param timestamp:
    :return:
    columns_with_few_unique_non_nan = {col: meta_with_clinical_from_tcga[col].dropna().unique() for col in meta_with_clinical_from_tcga.columns if (meta_with_clinical_from_tcga[col].nunique(dropna=True) < 10 ) & (meta_with_clinical_from_tcga[col].nunique(dropna=True) > 0 ) }

    """
    interested_meta_cols = ["diagnosis",
                            "tumor_stage", 'tumor_percent',
                            # "xml_lab_procedure_her2_neu_in_situ_hybrid_outcome_type",
                            # "xml_lab_proc_her2_neu_immunohistochemistry_receptor_status",
                            # "xml_her2_erbb_pos_finding_cell_percent_category"
                            ]

    vis_mode = "phate"
    s2_hoverkeys = ["diagnosis", "tumor_percent", "short_sample_id"]
    # Initialize the DatasetVisualizer
    visualizer = DatasetVisualizer(
            gex_clinical_from_tcga, num_genes, interested_meta_cols, save_dir, if_plot_eda=False)
    drug = "overall"
    if visualizer.if_plot_eda:
        # Visualize original data
        visualizer.visualize_original_data(drug, "all", vis_mode, timestamp)

        # Plot individual cancer projections
        visualizer.plot_individual_cancer_projection(drug)

        # Interactive Bokeh plots
        visualizer.interactive_bokeh_for_dgea(drug, timestamp, vis_mode,
                                              ["days_to_death", "binary_response", "disease_code",
                                               "short_sample_id"])

        # Interactive subplot plotting
        visualizer.interactive_plot_with_subplots_on_one_key(drug, timestamp)

        ## interactive plot with each subpolot color code by one meta-column
        visualizer.interactive_all_meta_subplots(drug, timestamp, vis_mode)


    # for aug_by_col in ["diagnosis"]:
    #     aug_gex, aug_label_dict = gexmix.augment_random(
    #             aug_by_col, target_features=gex_clinical_from_tcga.iloc[:, -num_genes:].values,
    #             target_label_dict=merged_label_dict,
    #             keys4mix=["diagnosis", ]
    #     )
    #
    #
    # for color_by, if_gradient in zip(["tumor_stage", "diagnosis"], [False, False]):
    #
    #     interactive_bokeh_with_select(
    #             projection[:, 0], projection[:, 1],
    #             hover_notions=hover_notion,
    #             table_columns=["x", "y"] + s2_hoverkeys,
    #             color_by=color_by, if_color_gradient=if_gradient,
    #             s2_hoverkeys=s2_hoverkeys,
    #             title=f"colorby {color_by}",
    #             mode="umap",
    #             postfix=f"{timestamp}-{color_by}-orig{len(projection)}",
    #             save_dir=save_dir, scatter_size=6
    #     )

def load_csv_perform_DGEA(filename, gene_df, default_save_name=None, title="drug", sample_identifier="short_sample_id", save_dir="./"):
    from scipy.stats import ttest_ind

    # Load the selected_data.csv file
    data = pd.read_csv(filename)

    # Split data into Selection A and B
    data_A = data[data['Selection'] == 'A']
    data_B = data[data['Selection'] == 'B']
    sample_id_A = data_A[sample_identifier].tolist()
    sample_id_B = data_B[sample_identifier].tolist()

    gene_data_A = gene_df[gene_df[sample_identifier].isin(sample_id_A)]
    gene_data_B = gene_df[gene_df[sample_identifier].isin(sample_id_B)]

    # For demonstration, let's assume 'x' and 'y' are gene expression values for two genes
    genes = list(gene_data_A.columns)[1:]
    # Calculate log fold change and p-values using t-test
    log_fold_changes = []
    p_values = []
    for gene in genes:
        # Calculate log fold change
        mean_A = gene_data_A[gene].mean()
        mean_B = gene_data_B[gene].mean()
        log_fold_change = np.log2(mean_B / mean_A)
        log_fold_changes.append(log_fold_change)
        # Perform t-test to compute p-value
        t_stat, p_val = ttest_ind(gene_data_A[gene], gene_data_B[gene], equal_var=False)
        p_values.append(p_val)
    # Adjust p-values (Benjamini-Hochberg FDR)
    p_adjusted = np.minimum(
            1, np.array(p_values) * len(p_values) / np.arange(
                    1, len(p_values) + 1))
    # Create a DataFrame with the results
    dgea_results = pd.DataFrame(
            {
                    'gene': genes,
                    'log_fold_change': log_fold_changes,
                    'p_value': p_values,
                    'p_adjusted': p_adjusted
            })
    # Compute -log10(p-value) for the volcano plot
    dgea_results['neglog10_p_value'] = -np.log10(dgea_results['p_value'])
    # Highlight significant genes
    significance_threshold = 0.05
    dgea_results['significant'] = dgea_results['p_adjusted'] < significance_threshold

    # Sort by 'significant' (descending), 'log_fold_change' (descending), 'neglog10_p_value' (descending)
    sorted_dgea_results = dgea_results.sort_values(
            by=['significant', 'log_fold_change', 'neglog10_p_value'],
            ascending=[False, False, False]
    )
    # Optionally, save the DGEA results to a CSV
    sorted_dgea_results.to_csv(path.join(save_dir, f"{default_save_name}-DGEA_results.csv"), index=False)

    # Plot a volcano plot using matplotlib and seaborn
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='log_fold_change', y='neglog10_p_value', data=dgea_results, s=20)
    sns.scatterplot(
            x='log_fold_change', y='neglog10_p_value', data=dgea_results[dgea_results['significant']],
            color='red', s=20, label='Significant Genes')
    # Add labels and a horizontal line for significance threshold
    plt.axhline(
            -np.log10(significance_threshold), ls='--', color='gray',
            label=f'p-value = {significance_threshold}')
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10(p-value)')
    plt.title('Volcano Plot of Differential Gene Expression')
    plt.legend()
    plt.tight_layout()
    # Save and display the plot
    plt.savefig('volcano_plot.png')
    plt.show()



    # visualize mix projection --> get interested clusters --> DGA
    hover_notion = []
    s2_hoverkeys = ["gene", "neglog10_p_value", "log_fold_change"]
    for key in s2_hoverkeys:
        hover_notion.append((key, dgea_results[key].values))

    interactive_bokeh_with_select_test_for_DGEA(
            dgea_results["log_fold_change"].values, dgea_results["neglog10_p_value"].values,
            hover_notions=hover_notion,
            height=500, width=500,
            key2separate="significant",
            if_multi_separate=False,
            key2color="neglog10_p_value",
            title=f"Volcano {title} ",
            table_columns=["x", "y"] + s2_hoverkeys,
            s2_hoverkeys=s2_hoverkeys,
            default_save_name=default_save_name,
            postfix=f"Volcano {path.basename(filename)}",
            if_color_gradient=True,
            save_dir=save_dir, scatter_size=6
    )
    return dgea_results


# 2024.09.05
def analyze_per_cancer(
        clinical_tcga_valid, drug_counts, gex_clinical_from_tcga, gexmix,
        meta_with_clinical_from_tcga, if_apply_augmentation=False, save_dir="./", timestamp="."
        ):
    """
    analyze the per drug visualization and effet of gexmix in drug response
    :param clinical_tcga_valid:
    :param drug_counts:
    :param gex_clinical_from_tcga:
    :param gexmix:
    :param meta_with_clinical_from_tcga:
    :param save_dir:
    :param timestamp:
    :return:
    """

    num_genes = gex_clinical_from_tcga.shape[1] - 3  ## first 3 are sample id, short_sample_id, diagnosis
    interested_meta_cols = ["days_to_death",
                            "primary_therapy_outcome_success",
                            "binary_response", "disease_code"]
    if "disease_code" not in meta_with_clinical_from_tcga.columns:
        meta_with_clinical_from_tcga.insert(1, "disease_code", meta_with_clinical_from_tcga["diagnosis"])
    # if "disease_code" not in gex_clinical_from_tcga.columns:
    #     gex_clinical_from_tcga.insert(1, "disease_code", gex_clinical_from_tcga["diagnosis"])
    merged_all_tcga = pd.merge(
            meta_with_clinical_from_tcga, gex_clinical_from_tcga,
            on=["sample_id", "short_sample_id"], how="left", suffixes=('', '_gex')
    )

    # crs_tcga_meta_data = tcga_dataset.meta_data[
    #     tcga_dataset.meta_data["diagnosis"].isin(["COAD", "READ"])]
    #
    # # Remove the suffix from 'sample_id' in df1 to match with 'Sample ID' in df2
    # crs_tcga_meta_data['sample_id_cleaned'] = crs_tcga_meta_data['sample_id'].str[
    #                                           :-1]  # Extract the part without the 'A' suffix
    #
    # # Merge the two dataframes based on the cleaned 'sample_id' in df1 and 'Sample ID' in df2
    # CRC_cms_clinical_fn = r"C:\Users\DiyuanLu\Downloads\coadread_tcga_clinical_data.tsv"
    # CRC_cms_clinical_data = pd.read_table(CRC_cms_clinical_fn, delimiter="\t")
    # merged_df = pd.merge(
    #     crs_tcga_meta_data, CRC_cms_clinical_data, left_on='sample_id_cleaned',
    #     right_on='Sample ID', how='inner')

    uniq_cancers = meta_with_clinical_from_tcga["disease_code"].unique()
    for ind, cancer in enumerate(list(uniq_cancers)): #uniq_cancers:
        print(f'{ind} {cancer}')
        cancer_specific_meta_from_Clinical = clinical_tcga_valid[
            clinical_tcga_valid["disease_code"] == cancer]

        # Combining all potential valid outcome info
        cancer_specific_meta_from_Clinical['primary_outcome'] = cancer_specific_meta_from_Clinical['primary_therapy_outcome_success'].combine_first(
                cancer_specific_meta_from_Clinical['measure_of_response'])

        cancer_clinical_dir = path.join(save_dir, cancer)
        if not path.exists(cancer_clinical_dir):
            makedirs(cancer_clinical_dir)

        merged_df2 = pd.merge(
                cancer_specific_meta_from_Clinical, merged_all_tcga,
                on="short_sample_id", how='left', suffixes=('', '_gex')
        )

        for vis_mode in ["umap", "phate"]:
            # # visualize original data for drug
            projection = visualize_data_with_meta(
                merged_df2.iloc[:, -num_genes:].values,
                merged_df2,
                interested_meta_cols,
                postfix=f"{cancer}-{vis_mode}-{timestamp}",
                cmap="jet", figsize=[8, 6], vis_mode=vis_mode,
                save_dir=cancer_clinical_dir
                )

            # Step 1: allow for subseting A and B for DGEA. visualize mix projection --> get interested clusters --> DGA
            hover_notion = []
            s2_hoverkeys = ["days_to_death", "drug_name", "binary_response", "disease_code", "short_sample_id"]
            for key in s2_hoverkeys:
                hover_notion.append((key, merged_df2[key]))

            interactive_bokeh_with_select_test_for_DGEA(
                    projection[:, 0], projection[:, 1],
                    hover_notions=hover_notion,
                    height=500, width=500,
                    key2color="binary_response",
                    key2separate="drug_name",
                    title=f"{cancer}", mode=vis_mode,
                    table_columns=["x", "y"] + s2_hoverkeys,
                    s2_hoverkeys=s2_hoverkeys,
                    postfix=f"{cancer}-{timestamp}AB-{vis_mode}",
                    if_color_gradient=True,
                    if_multi_separate=True,
                    save_dir=cancer_clinical_dir, scatter_size=6
            )
        print("ok")
        aug_mix_col = ["primary_outcome", "days_to_death", "disease_code",
                       "binary_response", "drug_name"]
        merged_label_df = merged_df2[["short_sample_id"] + aug_mix_col]

        ## keep all meta, the odd rows are backbones, the even rows are auxil. samples
        gexmix.num2generate = 400
        gene_names = merged_df2.columns[-num_genes:]
        for aug_by_col in ["primary_outcome", "binary_response"]:
            # Set a seed for reproducibility
            np.random.seed(99)
            timestamp = datetime.now().strftime("%m-%dT%H-%M")

            aug_gex, aug_label_dict = gexmix.augment_random(
                    aug_by_col, target_features=merged_df2.iloc[:, -num_genes:].values,
                    target_label_dict=merged_label_df,
                    keys4mix=aug_mix_col,
            )

            with open(
                    path.join(
                            cancer_clinical_dir,
                            f"{timestamp}_aug_{cancer}_by{aug_by_col[0:5]}_{len(aug_gex)}.pickle"
                    ), 'wb'
            ) as handle:
                aug_dict = {}
                aug_dict["gex_data"] = aug_gex
                aug_dict["meta_data"] = merged_df2.iloc[:, 0:-num_genes]
                pickle.dump(aug_dict, handle)

            # tag sample index for differential gene expresison analsyis
            aug_label_dict["sample_id"] = np.arange(len(aug_gex))
            for vis_mode in ["umap", "phate"]:
                aug_projection = visualize_data_with_meta(
                        aug_gex, pd.DataFrame(aug_label_dict),
                        aug_mix_col,
                        postfix=f"{cancer}-{timestamp}-AUG{gexmix.num2aug}-by-{aug_by_col[0:3]}-{vis_mode}",
                        cmap="jet", figsize=[8, 6], vis_mode="phate",
                        save_dir=save_dir
                )
                aug_label_dict = pd.DataFrame(aug_label_dict)

            print(f'{ind} {cancer} Done!')
            # # Extract the 'sample_id' column from the DataFrame (or dictionary) as a Series
            # sample_id_series = aug_label_dict["sample_id"]
            #
            # # Create the DataFrame by combining the 'sample_id' Series with the gene expression array (aug_gex)
            # gex_w_id = pd.DataFrame(data=aug_gex, columns=list(gene_names))
            #
            # # Add the 'sample_id' as the first column
            # gex_w_id.insert(0, 'sample_id', sample_id_series)
            # filename = r"C:\Users\DiyuanLu\Downloads\13-13T38-41-LGG-temodar-S-R.csv"
            # dgea_results = load_csv_perform_DGEA(filename, gex_w_id, default_save_name=path.basename(filename), title=f"{cancer}+{path.basename(filename)}",
            #                       sample_identifier="sample_id", save_dir=cancer_clinical_dir)
            #
            # enrichment_results = get_GSEA_with_gprofiler(dgea_results)


def get_GSEA_with_gprofiler(dgea_results):
    from gprofiler import GProfiler
    # Initialize the g:Profiler object
    gp = GProfiler(return_dataframe=True)
    gene_list = dgea_results["gene"]
    # Perform GSEA with g:Profiler
    results = gp.profile(organism='hsapiens', query=gene_list)
    # View results
    print(results)


def analyze_per_drug_tcga_clinical2(
    clinical_tcga_valid, drug_counts, gex_clinical_from_tcga, gexmix,
    meta_with_clinical_from_tcga, num_genes=2000, if_apply_augmentation=False, save_dir="./", timestamp="."
):
    """
    Analyze the per drug visualization and effect of gexmix in drug response.
    :param clinical_tcga_valid:
    :param drug_counts:
    :param gex_clinical_from_tcga:
    :param gexmix:
    :param meta_with_clinical_from_tcga:
    :param save_dir:
    :param timestamp:
    :return:
    """
    interested_meta_cols = ["days_to_death", "primary_therapy_outcome_success", "binary_response", "disease_code"]
    if "disease_code" not in meta_with_clinical_from_tcga.columns:
        meta_with_clinical_from_tcga.insert(1, "disease_code", meta_with_clinical_from_tcga["diagnosis"])

    # Filter in [all TCGA_meta with available clinical] with [drug_specific_meta in clinical data]
    merged_tcga = pd.merge(
        meta_with_clinical_from_tcga, gex_clinical_from_tcga,
        on=["sample_id", "short_sample_id"], how="left", suffixes=('', '_gex')
    )
    assert num_genes == gex_clinical_from_tcga.shape[1] - 3, "num_genes not correct"

    for ind in range(0, 100):
        drug = drug_counts["drug_name"][ind]
        drug_count = drug_counts["total_count"][ind]
        drug_specific_meta_from_clinical = clinical_tcga_valid[clinical_tcga_valid["drug_name"] == drug]

        if drug_count < 20:
            continue

        drug_clinical_dir = path.join(save_dir, drug)
        if not path.exists(drug_clinical_dir):
            makedirs(drug_clinical_dir)

        # Merge the resulting TCGA data with meta_clinic, using meta_clinic as the template
        merged_df2 = pd.merge(
            drug_specific_meta_from_clinical, merged_tcga,
            on="short_sample_id", how='left', suffixes=('', '_gex')
        )

        vis_mode = "umap"
        # Initialize the DatasetVisualizer
        visualizer = DatasetVisualizer(
            merged_df2, num_genes, interested_meta_cols, save_dir, if_plot_eda=False
        )

        if visualizer.if_plot_eda:
            # Visualize original data
            visualizer.visualize_original_data(drug, drug_count, vis_mode, timestamp)

            # Plot individual cancer projections
            visualizer.plot_individual_cancer_projection(drug)

            # Interactive Bokeh plots
            visualizer.interactive_bokeh_for_dgea(drug, timestamp, vis_mode, ["days_to_death", "binary_response", "disease_code", "short_sample_id"])

            # Interactive subplot plotting
            visualizer.interactive_plot_with_subplots_on_one_key(drug, timestamp)

            # Interactive plot with each subplot color-coded by one meta-column
            visualizer.interactive_all_meta_subplots(drug, timestamp, vis_mode)

        # Train and evaluate models
        model_collection = GDSCModelCollection(gexmix)
        model_collection.performance_metrics = {"ori": {}, "aug": {}}

        # Train and save models with original data
        model_collection.train_and_save_models(merged_df2, num_genes, drug, "ori", save_dir=drug_clinical_dir)

        coll_results_dict = []
        if if_apply_augmentation:
            aug_mix_col = ["primary_therapy_outcome_success", "days_to_death", "disease_code", "binary_response"]

            # Use augmentation to train TCGA clinical data for drug response prediction
            drug_results_df = overall_aug_mix_within_TCGA_investigation(
                aug_mix_col, drug_counts, gexmix, interested_meta_cols,
                merged_df2, merged_df2, num_genes=num_genes, drug=drug, cancer_type_column="disease_code", key2color="disease_code",
                save_dir=drug_clinical_dir, timestamp=timestamp, vis_mode=vis_mode
            )
            coll_results_dict.append(drug_results_df)
            if coll_results_dict:
                coll_results_df = pd.concat(coll_results_dict)
                coll_results_df.to_csv(path.join(path.dirname(drug_clinical_dir),
                                                 f"{timestamp}-overall_drp_{drug}.csv"))

            # Train and save models with augmented data
            model_collection.train_and_save_models(merged_df2, num_genes, drug, "aug", save_dir=drug_clinical_dir)

        # Plot overall performance
        model_collection.plot_overall_performance(model_collection.performance_metrics, prefix=drug, save_dir=drug_clinical_dir)



# 2024.09.05
def analyze_per_drug_tcga_clinical(
        clinical_tcga_valid, drug_counts, gex_clinical_from_tcga, gexmix,
        meta_with_clinical_from_tcga, num_genes=2000, if_apply_augmentation=False, save_dir="./", timestamp="."
        ):
    """
    analyze the per drug visualization and effet of gexmix in drug response
    :param clinical_tcga_valid:
    :param drug_counts:
    :param gex_clinical_from_tcga:
    :param gexmix:
    :param meta_with_clinical_from_tcga:
    :param save_dir:
    :param timestamp:
    :return:
    """

    interested_meta_cols = ["days_to_death",
                            "primary_therapy_outcome_success",
                            "binary_response", "disease_code"]
    if "disease_code" not in meta_with_clinical_from_tcga.columns:
        meta_with_clinical_from_tcga.insert(
            1, "disease_code", meta_with_clinical_from_tcga["diagnosis"])

    # filter in [all TCGA_meta with availabel clinical] with [durg_specific_meta in clinical data]
    merged_tcga = pd.merge(
            meta_with_clinical_from_tcga, gex_clinical_from_tcga,
            on=["sample_id", "short_sample_id"], how="left", suffixes=('', '_gex')
    )
    assert num_genes == gex_clinical_from_tcga.shape[1] - 4, "num_genes not correct"
    for ind in range(0, 20):
        drug = drug_counts["drug_name"][ind]
        drug_count = drug_counts["total_count"][ind]
        drug_specific_meta_from_clinical = clinical_tcga_valid[
            clinical_tcga_valid["drug_name"] == drug]

        if drug_count < 20:
            continue

        drug_clinical_dir = path.join(save_dir, drug)
        if not path.exists(drug_clinical_dir):
            makedirs(drug_clinical_dir)

        # Merge the resulting TCGA data with meta_clinic, using meta_clinic as the template
        merged_df2 = pd.merge(
                drug_specific_meta_from_clinical, merged_tcga,
                on="short_sample_id", how='left', suffixes=('', '_gex')
        )
        ## remove duplicates
        merged_df2 = merged_df2.groupby("short_sample_id").first().reset_index()
        # merged_df2.drop("DRUG_NAME", axis=1, inplace=True)  ## the same drug might have different names
        # filtered_merged_df2 = merged_df2[~merged_df2.duplicated(keep=False)]   ## still duplicates

        vis_mode = "umap"
        # Initialize the DatasetVisualizer
        visualizer = DatasetVisualizer(
            merged_df2, num_genes, interested_meta_cols, save_dir, if_plot_eda=False)

        if visualizer.if_plot_eda:
            # Visualize original data
            visualizer.visualize_original_data(drug, drug_count, vis_mode, timestamp)

            # Plot individual cancer projections
            visualizer.plot_individual_cancer_projection(drug)

            # Interactive Bokeh plots
            visualizer.interactive_bokeh_for_dgea(drug, timestamp, vis_mode, ["days_to_death", "binary_response", "disease_code", "short_sample_id"])

            # Interactive subplot plotting
            visualizer.interactive_plot_with_subplots_on_one_key(drug, timestamp)

            ## interactive plot with each subpolot color code by one meta-column
            visualizer.interactive_all_meta_subplots(drug, timestamp, vis_mode)

        ## Step 2: Load saved setA and setB for vol
        # gex_w_id = merged_df2[["short_sample_id"]+list(merged_df2.columns)[-num_genes:]]
        # filename = r"C:\Users\DiyuanLu\Downloads\11-19T07-09-docetaxel-BRCA-BRCA.csv"
        # load_csv_perform_DGEA(filename, gex_w_id, title=drug, save_dir=drug_clinical_dir)

        if if_apply_augmentation:
            aug_mix_col = ["binary_response", "primary_therapy_outcome_success",
                           "days_to_death", "disease_code"]

            ## use augmentation to train TCGA clinical data for drug response prediction
            overall_aug_mix_within_TCGA_investigation(
                aug_mix_col, drug_counts, gexmix, interested_meta_cols,
                merged_df2, merged_df2, num_genes=num_genes, drug=drug,
                    cancer_type_column="disease_code", key2color="disease_code",
                    save_dir=drug_clinical_dir, timestamp=timestamp, vis_mode=vis_mode)

            ## augment CCLE with TCGA -> train CCLE drug response w/wo DA -> predict on TCGA w/w DA
            ## TODO: train a transformer with all potential gene expression data with DA as encoder


def latent_expression_distribution_measure(projection, gene_expression_df):
    """
    compute the distribution of a given gene's expression level in the projection space for visualziation
    :param projection: 2d array [n_samples, 2]
    :param gene_expression_df: pd.Dataframe
    :return:
    """
    from scipy.stats import f_oneway, pearsonr
    from sklearn.metrics import mutual_info_score
    # from skimage.filters import sobel
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    # 1. Variance of Expression Across Binned Clusters in Latent Space
    def compute_variance_across_bins(latent_points, gene_expression, n_bins=10):
        bins = pd.cut(latent_points[:, 0], bins=n_bins, labels=False)
        variances = []
        for gene_idx in range(gene_expression.shape[1]):
            expression_by_bin = [gene_expression[bins == i, gene_idx] for i in range(n_bins)]
            variances.append(np.var([np.mean(bin_values) for bin_values in expression_by_bin]))
        return variances

    # 2. Correlation with Latent Dimensions
    def compute_correlation_with_latent(latent_points, gene_expression):
        correlations = []
        for gene_idx in range(gene_expression.shape[1]):
            gene_corr = [
                    pearsonr(latent_points[:, dim], gene_expression[:, gene_idx])[0]
                    for dim in range(latent_points.shape[1])
            ]
            correlations.append(np.mean(np.abs(gene_corr)))
        return correlations

    # 3. Entropy of Expression Levels
    # Modified Entropy Calculation with Min-Max Scaling
    def compute_entropy(expression_levels):
        # Rescale to [0, 1] range
        scaler = MinMaxScaler()
        scaled_expression = scaler.fit_transform(expression_levels.reshape(-1, 1)).flatten()

        # Normalize to a probability distribution
        p_data = scaled_expression / np.sum(scaled_expression)
        entropy = -np.sum(
            p_data * np.log(p_data + 1e-10))  # Adding a small constant to avoid log(0)
        return entropy

    def compute_expression_entropy(gene_expression):
        return [compute_entropy(gene_expression[:, gene_idx]) for gene_idx in
                range(gene_expression.shape[1])]

    # 4. Gradient Index Using Sobel Filter
    def compute_gradient_index(latent_points, gene_expression, grid_size=50):
        # Map points to a grid
        x_bins = pd.cut(latent_points[:, 0], bins=grid_size, labels=False)
        y_bins = pd.cut(latent_points[:, 1], bins=grid_size, labels=False)

        gradient_indices = []
        for gene_idx in range(gene_expression.shape[1]):
            # Create grid representation of the gene expression levels
            grid = np.full((grid_size, grid_size), np.nan)
            for i in range(len(x_bins)):
                if not np.isnan(x_bins[i]) and not np.isnan(y_bins[i]):
                    grid[x_bins[i], y_bins[i]] = gene_expression[i, gene_idx]

            # Fill NaNs with neighboring values
            grid = pd.DataFrame(grid).fillna(method='ffill').fillna(method='bfill').values
            gradient_image = sobel(grid)
            gradient_index = np.mean(gradient_image)
            gradient_indices.append(gradient_index)
        return gradient_indices

    # 5. Spatial Autocorrelation (Moran's I)
    def compute_morans_i(latent_points, gene_expression, n_neighbors=10):
        morans_i_values = []
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(latent_points)
        for gene_idx in range(gene_expression.shape[1]):
            distances, indices = nbrs.kneighbors(latent_points)
            w = np.exp(-distances)  # weight matrix based on distance
            mean_expression = np.mean(gene_expression[:, gene_idx])
            deviation = gene_expression[:, gene_idx] - mean_expression
            numerator = np.sum(w * deviation[indices] * deviation.reshape(-1, 1))
            denominator = np.sum(deviation ** 2)
            morans_i = len(gene_expression) / np.sum(w) * numerator / denominator
            morans_i_values.append(morans_i)
        return morans_i_values

    # Running all metrics for each gene and combining into a single DataFrame
    def compute_all_metrics(latent_points, gene_expression):
        metrics = {
                'variance_across_bins': compute_variance_across_bins(
                    latent_points, gene_expression),
                'correlation_with_latent': compute_correlation_with_latent(
                    latent_points, gene_expression),
                'expression_entropy': compute_expression_entropy(gene_expression),
                # 'gradient_index': compute_gradient_index(latent_points, gene_expression),
                'morans_i': compute_morans_i(latent_points, gene_expression)
        }
        return pd.DataFrame(metrics, index=[f'Gene_{i}' for i in range(gene_expression.shape[1])])

    # Example usage
    results = compute_all_metrics(projection, gene_expression_df.values)
    print(results)

    # Perform PCA to get the latent points
    pca = PCA(n_components=2)
    pca_points = pca.fit_transform(gene_expression_df.values)
    row, col = 6, 6
    for sort_by in ["variance_across_bins", "correlation_with_latent", "morans_i",
                    "expression_entropy"]:
        results.sort_values(sort_by, inplace=True, ascending=False)
        # Plot 8x8 subplots
        fig, axes = plt.subplots(row, col, figsize=(20, 20))
        flat_axes = axes.flatten()
        plt.suptitle(f"Sort by {sort_by}")
        fig.subplots_adjust(hspace=0.02, wspace=0.02)
        gene_names = gene_expression_df.columns

        for i, gene in enumerate(results.index[:36]):
            ax = flat_axes[i]
            scatter = ax.scatter(
                    pca_points[:, 0], pca_points[:, 1],
                    c=gene_expression_df[gene], cmap='viridis', s=8, alpha=0.7
            )
            ax.set_title(f'{gene}')
            ax.set_xticks([])
            ax.set_yticks([])

        # Add color bar for reference
        fig.colorbar(scatter, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1, aspect=40)
    plt.show()

    ## compare gene expression distribution over latent projection
    compare_latent_spread_highlight_genes(results)


def compare_latent_spread_highlight_genes(axes, df, results, scatter_with_labels):
    from adjustText import adjust_text
    # Adjust the scatter plot to display gene names next to points and avoid label collision
    df = results.copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Helper function to plot scatter with gene labels
    def scatter_with_labels(x, y, ax, xlabel, ylabel, highlight_genes):
        sns.scatterplot(
                x=x, y=y, hue=x.index.isin(highlight_genes),
                palette={True: (1, 0, 0, 0.5), False: (0, 0, 1, 0.06)},
                # Set alpha values for red and blue
                legend=False, ax=ax
        )
        # Add text annotations for each gene, with text adjustment
        texts = []
        for gene, x_val, y_val in zip(df.index, df["expression_entropy"], df["morans_i"]):
            color = "red" if gene in highlight_genes else "blue"
            texts.append(ax.text(x_val, y_val, gene, fontsize=8, color=color))

        # Adjust text to avoid overlapping
        adjust_text(
                texts, only_move={'points': 'y', 'text': 'y'}, iter_lim=10, time_lim=5,
                arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    percentile_90 = results.quantile(0.995)
    # Define the genes that exceed the 90th percentile in any metric
    highlight_genes = results[
        (results["variance_across_bins"] > percentile_90["variance_across_bins"]) |
        (results["correlation_with_latent"] > percentile_90["correlation_with_latent"]) |
        (results["expression_entropy"] > percentile_90["expression_entropy"]) |
        (results["morans_i"] > percentile_90["morans_i"])
        ].index
    # Plot each metric pair with adjusted text positions
    scatter_with_labels(
            df["variance_across_bins"], df["correlation_with_latent"], axes[0, 0],
            "Variance Across Bins", "Correlation with Latent", highlight_genes)
    axes[0, 0].set_title("Variance Across Bins vs Correlation with Latent")
    scatter_with_labels(
            df["variance_across_bins"], df["expression_entropy"], axes[0, 1],
            "Variance Across Bins", "Expression Entropy", highlight_genes)
    axes[0, 1].set_title("Variance Across Bins vs Expression Entropy")
    scatter_with_labels(
            df["correlation_with_latent"], df["morans_i"], axes[1, 0],
            "Correlation with Latent", "Moran's I", highlight_genes)
    axes[1, 0].set_title("Correlation with Latent vs Moran's I")
    scatter_with_labels(
            df["expression_entropy"], df["morans_i"], axes[1, 1],
            "Expression Entropy", "Moran's I", highlight_genes)
    axes[1, 1].set_title("Expression Entropy vs Moran's I")
    # Adjust layout for clarity
    plt.tight_layout()
    plt.show()


def overall_aug_mix_within_TCGA_investigation(aug_mix_col, drug_counts, gexmix, interested_meta_cols,
                                              merged_df2, merged_label_df, num_genes=5000, drug="drug", key2color="disease_code",
                                              cancer_type_column="disease_code",
                                              save_dir="./", timestamp="time", vis_mode="umap"):
    """

    :param aug_mix_col:
    :param drug:
    :param drug_clinical_dir:
    :param drug_counts:
    :param gexmix:
    :param interested_meta_cols:
    :param merged_data_df:
    :param merged_df2:
    :param merged_label_df:
    :param results_collect_df:
    :param timestamp:
    :param vis_mode:
    :return:
    """
    # augment the clinical data
    merged_data_df, mapping = gexmix.reencode_cancer_types(
            merged_df2, cancer_type_column=cancer_type_column
    )

    # Initialize DataFrame for results
    mean_metric_sum_df = pd.DataFrame(
            columns=["drug", "ori_or_aug", "model_name", "accuracy", "run", "num_samples",
                     "aug_by_col"])
    # for aug_by_col in aug_mix_col:  # , "binary_response", "primary_therapy_outcome_success"
    np.random.seed(949)

    ## prepare for ML model training
    drop_dup_nonan_x, drop_dup_nonan_y = process_data_handle_duplicates(merged_df2.iloc[:, -num_genes:],
                                                                        merged_df2.iloc[:, 0:-num_genes])
    check_GDSC_TCGA_col_stats(
            drop_dup_nonan_y, "binary_response", group_by_col='diagnosis', save_dir=save_dir,
            prefix=f"{drug}-aug_binary"
    )

    # X_train, df_y_train_dict, class_counts_str, X_test, df_y_test_dict = get_train_val_data(
    #         drop_dup_nonan_x, drop_dup_nonan_y)

    # Apply the cleaning function to the 'consensus_response' column
    drop_dup_nonan_y['consensus_response_cleaned'] = drop_dup_nonan_y['consensus_response'].apply(
            clean_consensus_response)

    # test the performance for classificaiton + data augmentation
    mean_metric_sum_df, results_w_prediction_gt_df = make_classifiers(
            gexmix,
            train_val_x=drop_dup_nonan_x,  ### when it is TCGA, this is overwritten
            train_val_y=drop_dup_nonan_y,  ### when it is TCGA, this is overwritten
            tcga_X_test=drop_dup_nonan_x,
            tcga_y_test=drop_dup_nonan_y,
            overall_results_df=mean_metric_sum_df,
            aug_by_col="binary_response",
            epochs=50,
            runs=1,
            model_names=["FNN", "linearregression", "elasticnet"],
            train_set_name="TCGA",
            test_set_name="TCGA",
            y_target_col="binary_response",  # aug_by_col and y_target_col don't need to be the same
            postfix=f"{drug}-aug_binary", drug=drug,
            regress_or_classify="regression",
            save_dir=save_dir
    )

    mean_metric_sum_df.to_csv(
            path.join(
                    path.dirname(save_dir),
                    f"{drug}-overall_drp_summary.csv"
            )
    )
    results_w_prediction_gt_df.to_csv(
            path.join(
                    path.dirname(save_dir),
                    f"{drug}-overall_drp_with_pred_gt_meta.csv"
            )
    )
    print("ok")
    return results_w_prediction_gt_df


def prepare_data_for_DGE_analysis(filenames_2group, meta_base, feature_base, col2match:str):
    """
    prepare data for differential gene expresison analysis
    :param filenames_2group:
    :param meta_base:
    :param feature_base:
    :param col2match:
    :return:
    """

    # differential gene expression analysis
    group1 = pd.read_csv(filenames_2group[0])
    group2 = pd.read_csv(filenames_2group[1])
    # Extract the group labels
    group1_meta = meta_base[meta_base[col2match].isin(group1[f' {col2match}'])]
    # Reset the index to get absolute row numbers
    group1_inds = np.where(meta_base[col2match].isin(group1[f' {col2match}']))[0]
    group2_meta = meta_base[meta_base[col2match].isin(group2[f' {col2match}'])]
    group2_inds = np.where(meta_base[col2match].isin(group2[f' {col2match}']))[0]
    # Subset the expression data based on the group labels
    group1_data = feature_base[group1_inds]
    group2_data = feature_base[group2_inds]

    return group1_data, group1_meta, group2_data, group2_meta

cancer_specific_meta2check = {
        "SARC": ["gdc_cases.diagnoses.vital_status", "cgc_case_tumor_status",
                           "cgc_case_new_tumor_event_after_initial_treatment",
                           "cgc_case_gender",
                           "cgc_slide_percent_lymphocyte_infiltration",
                           "cgc_slide_percent_monocyte_infiltration",
                           "cgc_slide_percent_neutrophil_infiltration",
                           "cgc_follow_up_tumor_status",
                           "xml_tissue_prospective_collection_indicator",
                           "xml_person_neoplasm_cancer_status",
                           "xml_primary_pathology_leiomyosarcoma_histologic_subtype",
                           "xml_primary_pathology_primary_tumor_lower_uterus_segment",
                           "xml_primary_pathology_tumor_depth",
                           "xml_primary_pathology_tumor_total_necrosis_percent",
                           "xml_primary_pathology_metastatic_site_at_diagnosis"
                           "cgc_case_new_tumor_event_after_initial_treatment",
                           ],
    "STAD":  [
            # "gdc_cases.demographic.race",
            # "gdc_cases.diagnoses.vital_status",
            # "gdc_cases.diagnoses.tissue_or_organ_of_origin",
            # "cgc_case_primary_therapy_outcome_success",
            # "cgc_case_tumor_status",
            # "cgc_case_vital_status",
            # "cgc_case_pathologic_n",
            "cgc_case_histological_diagnosis",
            "cgc_slide_percent_monocyte_infiltration",
            "cgc_slide_section_location",
            "cgc_radiation_therapy_radiation_therapy_site",
            "cgc_follow_up_tumor_status",
            "xml_tissue_prospective_collection_indicator",
            "xml_tissue_retrospective_collection_indicator",
            "xml_person_neoplasm_cancer_status",
            "xml_has_new_tumor_events_information",
            "xml_histological_type",
            "xml_anatomic_neoplasm_subdivision",
            "xml_neoplasm_histologic_grade",
            "xml_residual_tumor",
            "xml_primary_therapy_outcome_success",
            "xml_reflux_history",
            "xml_family_history_of_stomach_cancer",
        ],
    "BRCA": [
            "xml_lab_proc_her2_neu_immunohistochemistry_receptor_status",
            "xml_breast_carcinoma_estrogen_receptor_status",
            "xml_lab_procedure_her2_neu_in_situ_hybrid_outcome_type",
            "xml_her2_immunohistochemistry_level_result",
            "xml_immunohistochemistry_positive_cell_score",
            "xml_breast_carcinoma_progesterone_receptor_status",
            "xml_axillary_lymph_node_stage_method_type",
            "xml_number_of_lymphnodes_positive_by_ihc",
            "xml_histological_type",
            "xml_person_neoplasm_cancer_status",
            "cgc_case_histological_diagnosis",
            "cgc_sample_sample_type",
            "xml_breast_carcinoma_immunohistochemistry_pos_cell_score"
        ],
    "COAD": [
            "xml_loss_expression_of_mismatch_repair_proteins_by_ihc",
            "xml_colon_polyps_present",
            "xml_history_of_colon_polyps",
            "xml_kras_mutation_found",
            "xml_microsatellite_instability",
            "xml_perineural_invasion_present",
            "xml_lymphatic_invasion",
            "xml_primary_therapy_outcome_success",
            "xml_anatomic_neoplasm_subdivision",
            "xml_histological_type",
            "xml_person_neoplasm_cancer_status",
            "xml_race_list",
            "cgc_drug_therapy_pharmaceutical_therapy_type",
            "cgc_slide_section_location",
            "cgc_slide_percent_monocyte_infiltration",
            "cgc_case_icd10",
            "cgc_case_gender",
            "cgc_case_pathologic_t",
            "cgc_case_histological_diagnosis",
            "cgc_sample_sample_type",
            "gdc_cases.samples.portions.analytes.aliquots.concentration",
        ]
}


def investigate_skin_data_with_mixing(gexmix):

    timestamp = datetime.now().strftime("%d-%HT%M-%S")
    skin_save_dir = "../results/skin_data_analysis"
    num_gene = 5000
    if not path.exists(skin_save_dir):
        makedirs(skin_save_dir)
    # Augment skin data to varify
    skin_meta_data = pd.read_csv(
        "../data/data_christina/inflammatory_skin_clinical_annotations_col107.csv")
    skin_data = pd.read_csv(
        "../data/data_christina/inflammatory_skin_gene_expression_col17816.csv")
    filtered_skin_data = select_top_k_percent_std(skin_data, num_gene)
    skin_meta_data["Pattern"] = skin_meta_data["Pattern"].astype(str)
    # visualize the original data with meta
    skin_projection = visualize_data_with_meta(filtered_skin_data,
                                                        pd.DataFrame(skin_meta_data),
                                                        ["age", "Pattern","Sex_x",
                                                         "diag"],
                                                        postfix=f"-{timestamp}-skin{gexmix.num2aug}",
                                                        cmap="jet",
                                                        save_dir=skin_save_dir)
    # Convert DataFrame to dictionary with arrays
    skin_meta_data_dict = {col: skin_meta_data[col].to_numpy() for col in
                           skin_meta_data.columns}
    aug_skin_gex, aug_skin_label_dict = gexmix.augment_random("diag",
                                                              target_features=filtered_skin_data.values,
                                                              target_label_dict=skin_meta_data_dict,
                                                              keys4mix=["age", "Pattern",
                                                                        "Score", "Sex_x",
                                                                        "diag",
                                                                        "Response week 12"])
    aug_skin_label_dict["sample_id"] = np.arange(len(aug_skin_gex))
    aug_skin_projection = visualize_data_with_meta(aug_skin_gex,
                                                            pd.DataFrame(
                                                                aug_skin_label_dict),
                                                            ["age", "Pattern", "Score",
                                                             "Sex_x",
                                                             "diag", "Response week 12"],
                                                            postfix=f"-DiagAUG-Skin{gexmix.num2aug}",
                                                            cmap="jet",
                                                            save_dir=skin_save_dir)

    # save generated data and meta and projection
    with open(path.join(skin_save_dir, f"aug{len(skin_meta_data)}_{timestamp}.pickle"),
              'wb') as handle:
        aug_dict = {}
        aug_dict["gex_data_ori"] = filtered_skin_data
        aug_dict["meta_data_ori"] = pd.DataFrame(skin_meta_data)

        aug_dict["gex_data_aug"] = aug_skin_gex
        aug_dict["meta_data_aug"] = pd.DataFrame(aug_skin_label_dict)
        aug_dict["umap_aug"] = pd.DataFrame(aug_skin_projection)

        pickle.dump(aug_dict, handle)
    # plot individual projection of each cancer in skin data
    plot_individual_cancer_project(aug_skin_gex, aug_skin_label_dict, aug_skin_projection,
                                   key2group="Pattern", key2color="diag",
                                   prefix=f"{timestamp}-DiagAUG{gexmix.num2aug}")
    hover_notion = []
    aug_skin_label_dict["sample_id"] = np.arange(len(aug_skin_gex))
    s2_hoverkeys = ["diag", "Pattern", "sample_id"]
    for key in s2_hoverkeys:
        hover_notion.append((key, aug_skin_label_dict[key]))
    for color_by, if_gradient in zip(["Pattern", "diag"], [True, True]):
        interactive_bokeh_with_select(aug_skin_projection[:, 0], aug_skin_projection[:, 1],
                                      hover_notions=hover_notion,
                                      table_columns=["x", "y"] + s2_hoverkeys,
                                      color_by=color_by, if_color_gradient=if_gradient,
                                      s2_hoverkeys=s2_hoverkeys,
                                      title=f"Test color by {color_by}",
                                      mode="umap",
                                      postfix=f"{timestamp}-{color_by}-DiagAUG-skin-{len(aug_skin_gex)}",
                                      save_dir=skin_save_dir, scatter_size=6)

    # invetigate the bokeh clusters, select and save groups to investigate
    filenames_2group = [
        rf"{skin_save_dir}\selected+Test color by diag-03-00T05-24-gender1.txt",
        rf"{skin_save_dir}\selected+Test color by diag-03-00T05-24-gender0.txt"]
    differential_postfix = f"P4-gender1vs0"

    aug_skin_label_dict["sample_id"] = np.arange(len(aug_skin_gex))

    group1_data, group1_meta, group2_data, group2_meta = prepare_data_for_DGE_analysis(
        filenames_2group, pd.DataFrame(aug_skin_label_dict),
        aug_skin_gex, "sample_id")
    results = DGE_analysis([pd.DataFrame(group1_data, columns=tcga_dataset.gex_data.columns),
                            pd.DataFrame(group2_data, columns=tcga_dataset.gex_data.columns)],
                           [group1_meta, group2_meta], save_dir=skin_save_dir, p_val_col2use="",
                           postfix=f"skin-{differential_postfix}-{timestamp}")
    results['log2_fold_change'] = results['log2_fold_change'] - 1
    sorted_gene_list = results.sort_values(by=['p_value', 'log2_fold_change']).head(
        300)
    rrr = gene_set_enrichment_analysis(list(sorted_gene_list["gene"].values),
                                       key2use="log2_fold_change",
                                       postfix=f"{differential_postfix}",
                                       save_dir=skin_save_dir)


def visualize_group_normalized_samples_plot_multi_bokeh_AB(data_df, meta_df, prefix="CCLE",
                                       meta2check=["diagnosis", "primary_or_metastasis"],
                                       vis_mode="phate", key2color="diagnosis",
                                       key2separate="diagnosis",
                                       if_color_gradient=True,
                                       if_multi_separate=True,
                                       s2_hoverkeys=["sample_id", "diagnosis", "tumor_percent"],
                                       save_dir="./"):

    # # visualize original data for drug
    projection = visualize_data_with_meta(
            data_df.values,
            meta_df,
            meta2check,
            postfix=f"{prefix}-{vis_mode}",
            cmap="jet", figsize=[8, 6], vis_mode=vis_mode,
            save_dir=save_dir
    )
    # Step 1: allow for subseting A and B for DGEA. visualize mix projection --> get interested clusters --> DGA
    hover_notion = []
    for key in s2_hoverkeys:
        hover_notion.append((key, meta_df[key]))
    interactive_bokeh_with_select_test_for_DGEA(
            projection[:, 0], projection[:, 1],
            hover_notions=hover_notion,
            height=500, width=500,
            key2color=key2color,
            key2separate=key2separate,
            title=f"{prefix}", mode=vis_mode,
            table_columns=["x", "y"] + s2_hoverkeys,
            s2_hoverkeys=s2_hoverkeys,
            postfix=f"{prefix}-AB-{len(projection)}",
            if_color_gradient=if_color_gradient,
            if_multi_separate=if_multi_separate,
            save_dir=save_dir, scatter_size=6
    )


def investigate_different_normalization_2datasets(input_datasets, norm_method="combat", norm_based_cols=["dataset_name", "diagnosis"],
                                                  subset_based_col="diagnosis",
                                                  subset_to_extract=["BRCA", "LIHC"],
                                                  if_save_to_pickle=False,
                                                  save_dir="./"):
    """
    Investigate different normalization methods for a list of datasets.

    :param input_datasets: List of dataset objects. Each dataset should have `gex_data` and `meta_data` attributes.
    :param norm_method: Normalization method to be used (e.g., "combat").
    :param norm_based_cols: Columns to be used for normalization (e.g., batch labels).
    :param subset_based_col: Column to use for subsetting data.
    :param subset_to_extract: List of labels to extract from `subset_based_col`.
    :param if_norm_combined: Whether to normalize the combined dataset.
    :return: Normalized datasets or combined normalized dataset.
    """
    from combat.pycombat import pycombat
    def harmonize_data_given_col(feature_df, meta_data_df, vars_use=["diagnosis"]):
        """
        :param feature_df: df
        :param meta_data: df
        :param vars_use: list of meta data that need to be integrated
        :return:
        """
        import harmonypy as hm
        np.random.seed(42)
        data = feature_df.copy()

        # Convert data to numpy array for processing by harmonypy
        data_matrix = data.values

        # Run Harmony
        ho = hm.run_harmony(data_matrix, meta_data_df, vars_use, max_iter_harmony=20)

        # Extract the corrected embedding from the harmony object
        data_harmony_corrected = pd.DataFrame(ho.Z_corr.T, index=data.index, columns=data.columns)

        return data_harmony_corrected

    batch_dataset_lbs = []
    subset_data_list = []
    subset_meta_list = []
    shared_feature_col = []
    shared_meta_col = []
    normalized_datasets = []

    for ind, temp_dataset in enumerate(input_datasets):
        ## Get only subsets of the data for investigation
        sub_dataset, sub_dataset_meta = get_subset_data_given_labels(
                temp_dataset, label_col=subset_based_col, subset_to_extract=subset_to_extract)

        # Add a combined batch label to the metadata
        batch_dataset_lbs += [temp_dataset.dataset_name] * len(sub_dataset)

        if ind == 0:
            shared_feature_col = set(list(sub_dataset.columns))
            shared_meta_col = set(list(sub_dataset_meta.columns))
        else:
            shared_feature_col = shared_feature_col & set(list(sub_dataset.columns))
            shared_meta_col = shared_meta_col & set(list(sub_dataset_meta.columns))

        # # Step 1: Sample-wise z-score normalization for all samples in the dataset
        # df_sample_normalized = sub_dataset.apply(zscore, axis=1)

        # normalized_datasets.append((normalized_data, subset_meta))
        subset_data_list.append(sub_dataset)
        subset_meta_list.append(sub_dataset_meta)

    # If combining normalization is requested
    if len(input_datasets) > 1:
        # Find common genes and common metadata columns
        common_genes = set.intersection(*[set(ds.columns) for ds in subset_data_list])
        common_meta_cols = set.intersection(*[set(ds.columns) for ds in subset_meta_list])

        # Extract and concatenate shared gene expression and metadata columns
        combined_gex = pd.concat(
                [sub_data[list(common_genes)] for sub_data in subset_data_list], axis=0,
                ignore_index=True)
        combined_meta_shared = pd.concat(
                [sub_meta[list(common_meta_cols)] for sub_meta in subset_meta_list], axis=0,
                ignore_index=True)
        combined_meta = pd.concat(
                [sub_meta for sub_meta in subset_meta_list], axis=0, ignore_index=True)

        ### apply two step normalization
        # Step 1: Sample-wise z-score normalization for all samples in the dataset
        combined_gex_normed = combined_gex.apply(zscore, axis=1)
        new_combined_gex = combined_gex_normed.copy()
        new_combined_meta = combined_meta.copy()
        # new_combined_gex, new_combined_meta = two_step_normalization(combined_gex_normed, combined_meta)

        # Create a unified batch identifier
        new_combined_meta['combined_batch'] = new_combined_meta['dataset_name']  ##  + '_' + combined_meta['new_diagnosis'] #  + '_' + new_combined_meta['tumor_purity_bin']

        if norm_method == "harmony":
            combined_corrected_data = harmonize_data_given_col(
                    new_combined_gex, combined_meta_shared, vars_use=["combined_batch"])
        elif norm_method == "combat":
            combined_corrected_data = pycombat(data=new_combined_gex.T,
                                               batch=new_combined_meta['combined_batch'],
                                               mean_only=True).T
        else:
            raise ValueError("Given norm_method should be either harmony or combat")

    for ind, temp_dataset in enumerate(input_datasets):   ## all dataset keep a copy of the combined normalized
        # keep the combat corrected results
        temp_dataset.gex_data_combined = combined_corrected_data
        temp_dataset.meta_data_combined = combined_meta

    dataset_names_str = '+'.join([ele.dataset_name for ele in input_datasets])

    ## before normalization group-level normalization, plot bokeh with multi-select
    # visualize_group_normalized_samples_plot_multi_bokeh_AB(
    #         combined_gex_normed,
    #         combined_meta, prefix=f"combat-{dataset_names_str}-after-zscore-NO-tumor_purity_bin", vis_mode="umap",
    #         meta2check=["diagnosis", "dataset_name", "primary_or_metastasis",
    #                     "tumor_percent"],
    #         key2color="diagnosis",
    #         key2separate="diagnosis",
    #         if_color_gradient=False,
    #         if_multi_separate=True,
    #         s2_hoverkeys=["sample_id", "diagnosis", "dataset_name", "tumor_percent"],
    #         save_dir=save_dir)
    # visualize_group_normalized_samples_plot_multi_bokeh_AB(
    #         new_combined_gex,
    #         combined_meta, prefix=f"combat-{dataset_names_str}-after-2step-groupNorm-NO-tumor_purity_bin", vis_mode="umap",
    #         meta2check=["diagnosis", "dataset_name", "primary_or_metastasis",
    #                     "tumor_percent"],
    #         key2color="diagnosis",
    #         key2separate="diagnosis",
    #         if_color_gradient=False,
    #         if_multi_separate=True,
    #         s2_hoverkeys=["sample_id", "diagnosis", "dataset_name", "tumor_percent"],
    #         save_dir=save_dir)

    ## after group-level normalization, plot bokeh with multi-select
    visualize_group_normalized_samples_plot_multi_bokeh_AB(
            combined_corrected_data,
            combined_meta, prefix=f"combat-{dataset_names_str}-L1000-after-combat-onlysite", vis_mode="umap",
            meta2check=["diagnosis", "dataset_name", "primary_or_metastasis",
                        "tumor_percent"],
            key2color="diagnosis",
            key2separate="diagnosis",
            if_color_gradient=False,
            if_multi_separate=True,
            s2_hoverkeys=["sample_id", "diagnosis", "dataset_name", "tumor_percent"],
            save_dir=save_dir)

    if if_save_to_pickle:
        combined_norm_data = {}
        combined_norm_data["rnaseq"] = combined_corrected_data
        combined_norm_data["meta"] = combined_meta
        combined_norm_data["meta_shared"] = combined_meta_shared
        saved_pickle_name = path.join(save_dir,
                                      f"combat_onlysite_"
                                      f"{dataset_names_str}_dataset"
                                      f"_{len(combined_norm_data)}.pickle")
        with open(saved_pickle_name, "wb") as f:
            pickle.dump(combined_norm_data, f)

    ### if want to visualize the data and differential immediatly, you will need the following code
    # sub_2datasetes_meta_reset = sub_2datasetes_meta.reset_index(drop=True)
    # combat_harmonized_data_reset = combat_harmonized_data.reset_index(drop=True)
    #
    # combined_datadf = pd.concat(
    #         [sub_2datasetes_meta_reset["sample_id"], combat_harmonized_data_reset], axis=1)
    # load_csv_perform_DGEA(
    #         r"C:\Users\DiyuanLu\Downloads\10-02T03-25-54-Combat-2Sets-ESCA-upper-lower.csv",
    #         combined_datadf, sample_identifier="sample_id", title="Combat-2Sets-ESCA",
    #         save_dir=args.save_dir)
    return combined_corrected_data, combined_meta



def get_subset_data_given_labels(dataset_class, label_col="diagnosis", subset_to_extract=["BRCA", "LIHC"]):
    """

    :param dataset_class:
    :param label_col:
    :param labels_to_extract: ["BRCA", "LIHC"] or ["all"]
    :return:
    """
    # Apply the condition to get the desired rows with labels in the given list
    if len(subset_to_extract) == 1 and "all" == subset_to_extract[0]:
        # Select all rows
        condition = dataset_class.meta_data[label_col].notna()
    else:
        # Select rows based on the labels in labels_to_extract
        condition = dataset_class.meta_data[label_col].isin(subset_to_extract)

    # Filter the data based on the condition
    sub_features_df = dataset_class.gex_data[condition]
    sub_meta_df = dataset_class.meta_data[condition]

    return sub_features_df, sub_meta_df


def initialize_datasets(config, filename_dict, save_dir="./"):
    def initialize_dataset(dataset_name, filename_dict, dataset_type, save_dir="./", verbose=True):
        """
        Helper function to initialize a GExDataset, load its metadata, and load its gene
        expression data.
        """
        dataset = GExDataset(
                filename_dict[dataset_name]["filename"],
                filename_dict[dataset_name]["meta_filename"],
                dataset_type,
                verbose=verbose,
                save_dir=save_dir
        )
        dataset.load_gex_data()
        dataset.load_meta_data()

        if "TCGA" == dataset_name:
            dataset.meta_data.insert(
                    0,
                    "short_sample_id",
                    ["-".join(ele.split("-")[0:-1]) for ele in dataset.meta_data["sample_id"]],
            )

        return dataset

    datasets = {}
    for name, options in config.items():
        if options["enabled"]:
            dataset = initialize_dataset(name, filename_dict, name, save_dir=save_dir)
            datasets[name] = dataset
    return datasets


def copy_save_all_files(save_dir, src_dir):
    """
	Copy and save all files related to model directory
	:param args:
	:return:
	"""
    import shutil

    save_dir = path.join(save_dir, 'source_files')
    if not path.exists(save_dir):  # if subfolder doesn't exist, should make the directory and then save file.
        makedirs(save_dir)

    for item in listdir(src_dir):
        s = path.join(src_dir, item)
        d = path.join(save_dir, item)
        if path.isdir(s):
            shutil.copytree(s, d, ignore=shutil.ignore_patterns('*.pyc', 'tmp*', "*.h"))
        else:
            shutil.copy2(s, d)
    print('Done WithCopy File!')


if __name__ == "__main__":

    yaml_file = "/pan_parameters.yaml"

    # load parameters
    save_dir = "../results/GDSC-RES"
    if not path.exists(save_dir):  # if subfolder doesn't exist, should make the directory and then save file.
        makedirs(save_dir)

    filename_dict = get_filename_dict()

    dataset_config = {
            "TCGA": {
                    "enabled": False,  # True False Set to False if you want to exclude this dataset
                    "mix_config": {
                            "if_include_original": True,
                            "num2aug": 200,
                            "num2mix": 2,
                            "beta_param": 2,
                            "if_use_remix": True,
                            "if_stratify_mix": True,
                    },
            },
            "CCLE": {
                    "enabled": False,  # True  False Disable this dataset
                    "mix_config": {
                            "if_include_original": True,
                            "num2aug": 200,
                            "num2mix": 2,
                            "beta_param": 2,
                            "if_use_remix": True,
                            "if_stratify_mix": True,
                    },
            },
            "GDSC": {
                    "enabled": False,  # True  False Disable this dataset
                    "mix_config": {
                            "if_include_original": True,
                            "num2aug": 200,
                            "num2mix": 2,
                            "beta_param": 2,
                            "if_use_remix": True,
                            "if_stratify_mix": True,
                    },
            },
            "combat_3": {
                    "enabled": True, # True  False Disable this dataset
                    "mix_config": {
                            "if_include_original": True,
                            "num2aug": 200,
                            "num2mix": 2,
                            "beta_param": 1,
                            "if_use_remix": True,
                            "if_stratify_mix": True,
                    },
            },
    }

    # Initialize datasets
    datasets = initialize_datasets(dataset_config, filename_dict, save_dir=save_dir)

    # Apply GExMix based on configuration
    mix_objects = {}
    for name, options in dataset_config.items():
        if options["enabled"] and options["mix_config"] is not None:
            mix_objects[name] = GExMix(datasets[name], **options["mix_config"])

    # Example usage for combat_3 mix
    # combat_3_mix = mix_objects.get("combat_3")

    # Optional: Investigate normalization if enabled
    if_try_more_normalization = False ## False True # Toggle this
    if if_try_more_normalization:
        combined_corrected_data, combined_meta = investigate_different_normalization_2datasets(
                [datasets[name] for name in ["TCGA", "GDSC", "CCLE"] if name in datasets],
                norm_based_cols=["dataset_name", "diagnosis"],
                subset_based_col="diagnosis",
                subset_to_extract=["all"],
                norm_method="combat",
                if_save_to_pickle=False,
                save_dir=save_dir,
        )
    ### load GDSC and compare data augmentation to the DRP
    combat_3datasets = datasets["combat_3"]
    combat_3datasets_mix = mix_objects.get("combat_3")

    # # ### Part 1: train with GDSC validation on GDSC
    GDSC_DRP_with_mix_with_GDSC(
        combat_3datasets, combat_3datasets_mix,
        save_dir=save_dir)

    ### Part 2: train with GDSC/GDSC+TCGA validation on GDSC or TCGA
    GDSC_DRP_with_mix_with_TCGA(combat_3datasets, combat_3datasets_mix, filename_dict["TCGA"]["drug_response"])
    # #
    # # ### Part 3: train with TCGA validation on TCGA
    # clinical_tcga_investigation(combat_3datasets, combat_3datasets_mix,
    #                             filename_dict["TCGA"]["drug_response"],
    #                             save_dir=save_dir)
    print("ok")

"""
drug list from: Targets
Nilotinib: 1013
Temozolomide: 1375
Sorafenib: 30
Cisplatin: 1005
Paclitaxel: 11
Doxorubicin: 1386
Crizotinib: 37
Tanespimycin: 1026
PHA-665752: 6
Lapatinib: 119
Nutlin-3: 1047 
Saracatinib: 38
Crizotinib: 37
Panobinostat: 438
Sorafenib: 30
PD0325901: 1060
Palbociclib: 1054
Paclitaxel: 11
Selumetinib: 1498
PLX-4720: 1371
NVP-TAE684: 35

drugs from CODE-AE
AZD8055: 1059
BMS-754807: 184
Vorinostat: 1012
Temozolomide: 1375
Sunitinib: 5
Temozolomide: 1375
Sorafenib: 30
Nilotinib: 1013
Axitinib: 1021
Tamoxifen: 1199
Paclitaxel: 11
Doxorubicin: 1386
Cisplatin: 1005
Gemcitabine: 1393
PF-4708671: 1129
AZD7762: 1402
AZD6482: 1066
AZ628: 29
RO-3306: 1052
GSK269962A: 1192
GSK1904529A: 202
BMS-536924: 1091
Embelin: 172
JQ1: 1218
KU-55933: 1030
MK-2206: 1053
Bosutinib: 1019
CHIR-99021: 1241
PAC-1: 175
BI-2536: 60
PF-562271: 158
"""

