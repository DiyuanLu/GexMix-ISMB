import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from os import path, makedirs
from glob import glob
from adjustText import adjust_text
from scipy.stats import spearmanr, pearsonr, ttest_ind, wilcoxon, ttest_rel
from plotting_utils import generate_dark_colors
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix,
                             mean_squared_error, r2_score, roc_auc_score)



# Function to process a file and compute metrics
def process_file(file_path):
    df = pd.read_csv(file_path)
    metrics = compute_metrics(df["Ground Truth"], df["Predictions"])
    drug_name = path.basename(file_path).split('-')[1]
    metrics["Drug"] = drug_name
    return metrics


def determine_file_type(file_path):
    return "ori" if "ori" in file_path else "aug"


# Modified function to process a file and include file type
def process_file_with_type(file_path):
    df = pd.read_csv(file_path)
    metrics = compute_metrics(df["Ground Truth"], df["Predictions"])
    drug_name = path.basename(file_path).split('-')[1]
    file_type = determine_file_type(file_path)
    metrics["Drug"] = drug_name
    metrics["File Type"] = file_type
    return metrics



def load_csv_and_compute_and_plot_volcano_drug_level():
    # Get all files in the current directory
    files = glob(
            r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active projects\GexMix-ISMB\results\GDSC-RES\models\leave-cell-line-out\12-13T02-01_beta5.0_6clsZ\No*.csv")
    # Rewriting the process to calculate p-values and generate a volcano plot correctly
    # Load all ori and aug files
    ori_files = [file for file in files if 'ori' in file]
    aug_files = [file for file in files if 'aug' in file]
    # Initialize a list to store the metrics for the volcano plot
    volcano_plot_data = []
    # Process each pair of ori and aug files
    for ori_file, aug_file in zip(ori_files, aug_files):
        # Read the data
        ori_df = pd.read_csv(ori_file)
        aug_df = pd.read_csv(aug_file)

        # Extract the drug name
        drug_name = path.basename(ori_file).split('-')[1]

        # Compute performance metrics for ori and aug
        ori_metrics = compute_metrics(ori_df['Ground Truth'], ori_df['Predictions'])
        aug_metrics = compute_metrics(aug_df['Ground Truth'], aug_df['Predictions'])

        # Calculate LogFC for each metric
        for metric in ori_metrics.keys():
            logfc = np.log2(aug_metrics[metric] / ori_metrics[metric]) if ori_metrics[
                                                                              metric] != 0 else np.nan

            # Perform t-test on ori and aug predictions
            t_stat, p_value = ttest_ind(
                    ori_df['Predictions'], aug_df['Predictions'], equal_var=False)

            # Store results
            volcano_plot_data.append(
                    {
                            'Drug': drug_name,
                            'Metric': metric,
                            'LogFC': logfc,
                            'p-value': p_value
                    })
    # Convert the results into a DataFrame
    volcano_df = pd.DataFrame(volcano_plot_data)
    volcano_df['-log10(p-value)'] = -np.log10(volcano_df['p-value'])
    # Generate a volcano plot for each metric
    unique_metrics = volcano_df['Metric'].unique()
    for metric in unique_metrics:
        plt.figure(figsize=(8, 6))
        metric_data = volcano_df[volcano_df['Metric'] == metric]
        plt.scatter(
                metric_data['LogFC'],
                metric_data['-log10(p-value)'],
                c='b', alpha=0.7, label='Drugs'
        )
        plt.axhline(
                y=-np.log10(0.05), color='r', linestyle='--',
                label='Significance Threshold (p=0.05)')
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.6)
        plt.xlabel('Log2 Fold Change')
        plt.ylabel('-log10(p-value)')
        plt.title(f'Volcano Plot of {metric}\nOri vs Aug Predictions')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.close()



def plot_volcano_with_text_adjust(volcano_group_filtered_df, metric, logfc_cutoff=1.5, logp_cutoff=2):
    """
    Plot a volcano plot with significance cutoffs for LogFC and -log10(p-value), using text adjustment.

    Parameters:
        volcano_group_filtered_df (pd.DataFrame): DataFrame containing the volcano plot data.
        metric (str): The metric to plot (e.g., 'Spearman Correlation').
        logfc_cutoff (float): The cutoff for Log Fold Change significance.
        logp_cutoff (float): The cutoff for -log10(p-value) significance.

    Returns:
        None
    """
    # Filter the data for the selected metric
    metric_data = volcano_group_filtered_df[volcano_group_filtered_df['Metric'] == metric]

    # Determine significance based on cutoffs
    metric_data['Significant'] = (
        (metric_data['LogFC'] > logfc_cutoff) | (metric_data['LogFC'] < -logfc_cutoff)
    ) & (metric_data['-log10(p-value)'] > logp_cutoff)

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(
        metric_data['LogFC'],
        metric_data['-log10(p-value)'],
        c=metric_data['Significant'].map({True: 'blue', False: 'gray'}),
        alpha=0.7, label='Cancer Groups'
    )
    plt.axhline(y=logp_cutoff, color='r', linestyle='--', label=f'Significance Threshold (-logP={logp_cutoff})')
    plt.axvline(x=logfc_cutoff, color='r', linestyle='--', label=f'LogFC={logfc_cutoff}')
    plt.axvline(x=-logfc_cutoff, color='r', linestyle='--', label=f'LogFC={-logfc_cutoff}')
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-log10(p-value)')
    plt.title(f'Volcano Plot of {metric}\nDrug-Cancer Group Level with Cutoffs')

    # Annotate significant points using text adjustment
    texts = []
    for _, row in metric_data[metric_data['Significant']].iterrows():
        text = plt.text(
            row['LogFC'], row['-log10(p-value)'],
            f"{row['Drug']}-{row['Cancer Group']}",
            fontsize=8, alpha=0.8
        )
        texts.append(text)
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    plt.legend()
    plt.grid(alpha=0.3)
    plt.close()


def plot_overall_pred_vs_gt_scatter_with_contour(data_dirs):
    from scipy.stats import gaussian_kde

    for data_dir in data_dirs:
        beta_str = path.basename(data_dir).split('_')[-2]
        ori_data, aug_data = load_saved_pred_gt_files(data_dir, file_patten="No*.csv")
        density_dict = {}
        uniq_models = ori_data["model_name"].unique()

        for model_name, group_data in uniq_models:
            # Separate ori and aug cases
            ori_data = group_data[group_data["model_name"] == model_name]
            aug_data = group_data[group_data["model_name"] == model_name]

            # Create subplots
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(f"Predictions vs Ground Truth for {model_name}", fontsize=16)
            density_dict = {}
            # Plot for ori case
            for ax, case, case_data in zip(axes, ["ori", "aug"], [ori_data, aug_data]):
                if case_data.empty:  # Skip empty cases
                    ax.set_title(f"No Data ({case})")
                    ax.axis("off")
                    continue

                pearson_corr, _ = pearsonr(case_data["Ground Truth"], case_data["Predictions"])
                sp_corr, p_value = spearmanr(case_data["Ground Truth"], case_data["Predictions"])
                r2 = r2_score(case_data["Ground Truth"], case_data["Predictions"])
                corr_str = f"Pearson r: {pearson_corr:.2f}\nSpearman r: {sp_corr:.2f}\nR2:{r2:.2f}"

                # Calculate point density
                xy = np.vstack([case_data["Ground Truth"], case_data["Predictions"]])
                density_dict[case] = gaussian_kde(xy)(xy)
                # Create scatter plot with density-based coloring

                sns.scatterplot(
                    case_data,
                    x="Ground Truth",
                    y="Predictions",
                    c=density_dict[case],
                    cmap='viridis',
                    s=20,
                    ax=ax,
                    edgecolor='gray'
                    )
                ax.text(
                        0.03, 0.95, corr_str,
                        transform=ax.transAxes, fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
                )
                # Add diagonal line for perfect predictions
                min_val = min(case_data["Ground Truth"].min(), case_data["Predictions"].min())
                max_val = max(case_data["Ground Truth"].max(), case_data["Predictions"].max())
                ax.plot(
                        [min_val, max_val], [min_val, max_val], color='k', linestyle='--',
                        label='Perfect Prediction')
                # ax.set_colorbar(scatter, label='Density')
                ax.set_title(f"Prediction vs Ground Truth ({prefix})")
                ax.set_xlabel("Ground Truth")
                ax.set_ylabel("Predictions")
                ax.grid(alpha=0.3)
            plt.savefig(
                path.join(data_dir, f"0-{prefix}-{model_name}-all_scatter_with_density.png"))
        plt.close()
        # for case, case_data in zip(["ori", "aug"], [ori_data, aug_data]):
        #     if case == "aug":
        #         prefix = "DA with " + beta_str
        #     else:
        #         prefix = "No DA"
        #
        #     pearson_corr, _ = pearsonr(case_data["Ground Truth"], case_data["Predictions"])
        #     sp_corr, p_value = spearmanr(case_data["Ground Truth"], case_data["Predictions"])
        #     r2 = r2_score(case_data["Ground Truth"], case_data["Predictions"])
        #     corr_str = f"Pearson r: {pearson_corr:.2f}\nSpearman r: {sp_corr:.2f}\nR2:{r2:.2f}"
        #
        #     # Calculate point density
        #     xy = np.vstack([case_data["Ground Truth"], case_data["Predictions"]])
        #     density_dict[case] = gaussian_kde(xy)(xy)
        #     # Create scatter plot with density-based coloring
        #     plt.figure(figsize=(8, 6))
        #     scatter = plt.scatter(
        #             case_data["Ground Truth"],
        #             case_data["Predictions"],
        #             c=density_dict[case],
        #             cmap='viridis',
        #             s=20,
        #             edgecolor='gray'
        #     )
        #     plt.text(
        #             0.03, 0.95, corr_str,
        #             transform=plt.gca().transAxes, fontsize=15, verticalalignment='top',
        #             bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
        #     )
        #     # Add diagonal line for perfect predictions
        #     min_val = min(case_data["Ground Truth"].min(), case_data["Predictions"].min())
        #     max_val = max(case_data["Ground Truth"].max(), case_data["Predictions"].max())
        #     plt.plot(
        #             [min_val, max_val], [min_val, max_val], color='k', linestyle='--',
        #             label='Perfect Prediction')
        #     plt.colorbar(scatter, label='Density')
        #     plt.title(f"Prediction vs Ground Truth ({prefix})")
        #     plt.xlabel("Ground Truth")
        #     plt.ylabel("Predictions")
        #     plt.grid(alpha=0.3)
        #     plt.savefig(path.join(data_dir, f"0-{prefix}-all-{case}_scatter_with_density.png"))
        #     plt.close()


def load_saved_pred_gt_files_old(data_dir, file_patten="No*.csv"):
    files = glob(path.join(data_dir, file_patten))
    # Rewriting the process to calculate p-values and generate a volcano plot correctly
    # Load all ori and aug files
    ori_files = [file for file in files if 'ori' in file]
    aug_files = [file for file in files if 'aug' in file]
    ori_data = []
    aug_data = []
    # Process ori files
    for file in ori_files:
        data = pd.read_csv(file)
        data['ori_or_aug'] = 'ori'  # Assign 'ori' for original files
        ori_data.append(data)
    # Process aug files
    for file in aug_files:
        data = pd.read_csv(file)
        data['ori_or_aug'] = 'aug'  # Assign 'aug' for augmented files
        aug_data.append(data)

    return pd.concat(ori_data), pd.concat(aug_data)


def compute_metrics(ground_truth, predictions):
    """
    Compute performance metrics between ground truth and predictions.
    Parameters:
        ground_truth (pd.Series): Ground truth values.
        predictions (pd.Series): Predicted values.
    Returns:
        dict: Dictionary containing the computed metrics.
    """
    if len(ground_truth) >= 2:
        spearman_corr, _ = spearmanr(ground_truth, predictions)
        pearson_corr, _ = pearsonr(ground_truth, predictions)
        r2 = r2_score(ground_truth, predictions)
        rmse = mean_squared_error(ground_truth, predictions, squared=False)
    else:
        spearman_corr, pearson_corr, r2, rmse = np.nan, np.nan, np.nan, np.nan
    return {
            "Spearman Correlation": spearman_corr,
            "Pearson Correlation": pearson_corr,
            "R2 Score": r2,
            "RMSE": rmse
    }


def load_csv_and_compute_and_plot_volcano_group_level(data_dir, prefix=""):
    """
    Load the saved CSV files and compute performance metrics for each cancer group, then generate a volcano plot.
    :param data_dir:
    :param prefix:
    :return:
    """
    files = glob(path.join(data_dir, "No*.csv"))
    # Load all ori and aug files
    ori_files = [file for file in files if 'ori' in file]
    aug_files = [file for file in files if 'aug' in file]
    volcano_plot_group_data = []
    # Process each pair of ori and aug files
    for ori_file, aug_file in zip(ori_files, aug_files):
        # Read the data
        ori_df = pd.read_csv(ori_file)
        aug_df = pd.read_csv(aug_file)
        # Extract the drug name
        drug_name = path.basename(ori_file).split('-')[1]
        # Group by cancer type and compute performance metrics
        ori_grouped = ori_df.groupby('diagnosis').apply(
                lambda group: compute_metrics(group['Ground Truth'], group['Predictions'])
        )
        aug_grouped = aug_df.groupby('diagnosis').apply(
                lambda group: compute_metrics(group['Ground Truth'], group['Predictions'])
        )
        # Calculate LogFC and p-values for each cancer group
        for diagnosis in ori_grouped.index:
            if diagnosis in aug_grouped.index:
                # Ensure sufficient samples
                ori_metrics = ori_grouped.loc[diagnosis]
                aug_metrics = aug_grouped.loc[diagnosis]
                if not ori_metrics or not aug_metrics:
                    continue

                for metric in ori_metrics.keys():
                    logfc = np.log2(aug_metrics[metric] / ori_metrics[metric]) if ori_metrics[
                                                                                      metric] != 0 else np.nan
                    # Perform t-test between ori and aug predictions for this group
                    ori_values = ori_df[ori_df['diagnosis'] == diagnosis]['Predictions']
                    aug_values = aug_df[aug_df['diagnosis'] == diagnosis]['Predictions']
                    print(
                            f"{diagnosis}: ori_values= {ori_values.shape}, ori_metrics={aug_values.shape}")
                    t_stat, p_value = ttest_ind(ori_values, aug_values, equal_var=False)
                    # Store results
                    volcano_plot_group_data.append(
                            {
                                    'Drug': drug_name,
                                    'Cancer Group': diagnosis,
                                    'Metric': metric,
                                    'LogFC': logfc,
                                    'p-value': p_value
                            }
                    )

    # Convert results into a DataFrame
    volcano_group_df = pd.DataFrame(volcano_plot_group_data)
    volcano_group_df['-log10(p-value)'] = -np.log10(volcano_group_df['p-value'])

    logfc_cutoff = 2
    logp_cutoff = 2
    # Generate a volcano plot for each metric
    unique_metrics = volcano_group_df['Metric'].unique()

    for metric in unique_metrics:
        plt.figure(figsize=(10, 6))
        metric_data = volcano_group_df[volcano_group_df['Metric'] == metric]
        # Determine significance based on cutoffs
        metric_data['Significant'] = (
                                             (metric_data['LogFC'] > logfc_cutoff) | (
                                             metric_data['LogFC'] < -logfc_cutoff)
                                     ) & (metric_data['-log10(p-value)'] > logp_cutoff)

        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(
                metric_data['LogFC'],
                metric_data['-log10(p-value)'],
                c=metric_data['Significant'].map({True: 'blue', False: 'gray'}),
                alpha=0.7, label='Cancer Groups'
        )
        plt.axhline(
                y=logp_cutoff, color='r', linestyle='--',
                label=f'Significance Threshold (-logP={logp_cutoff})')
        plt.axvline(x=logfc_cutoff, color='r', linestyle='--', label=f'LogFC={logfc_cutoff}')
        plt.axvline(x=-logfc_cutoff, color='r', linestyle='--', label=f'LogFC={-logfc_cutoff}')
        plt.xlabel('Log2 Fold Change')
        plt.ylabel('-log10(p-value)')
        plt.title(f'Volcano Plot of {metric}\nDrug-Cancer Group Level with Cutoffs')

        # Annotate significant points using text adjustment
        texts = []
        for _, row in metric_data[metric_data['Significant']].iterrows():
            text = plt.text(
                    row['LogFC'], row['-log10(p-value)'],
                    f"{row['Drug']}-{row['Cancer Group']}",
                    fontsize=8, alpha=0.8
            )
            texts.append(text)
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(path.join(data_dir, f"{prefix}_volcano_plot_{metric}.png"))
        plt.show()
        plt.close()


def adjust_gt_col_name(case_data):
    if "Ground Truth" in case_data.columns:
        gt_col_name = "Ground Truth"
    elif "ground_truth" in case_data.columns:
        gt_col_name = "ground_truth"

    if "Predictions" in case_data.columns:
        pred_col_name = "Predictions"
    elif "Prediction" in case_data.columns:
        pred_col_name = "Prediction"
    elif "predictions" in case_data.columns:
        pred_col_name = "predictions"
    elif "prediction" in case_data.columns:
        pred_col_name = "prediction"

    return gt_col_name, pred_col_name

def process_metrics_for_each_file_one_folder(data_dir, file_pattern, param_extract_func, extract_label="beta",
                                            if_with_clf_metrics=False, return_raw_data=False, groupby_col=["Drug", "model_name"]):
    """
    Process a single directory and compute metrics for ori and aug data under different hyperparameters.

    Parameters:
        data_dir (str): Directory containing prediction files.
        file_pattern (str): Pattern to match prediction files.
        extract_label (str): Label to extract hyperparameter from directory name.
        param_extractor (callable): Function or lambda to extract the hyperparameter from the directory name.
        groupby_col (callable): should almost always have model_name and Drug

    Returns:
        list: Processed data for ori and aug combined.
    """
    result_data = []
    ori_data, aug_data = load_saved_pred_gt_files(data_dir, file_pattern=file_pattern)
    extract_label_value = param_extract_func(data_dir)

    for case, case_data in zip(["ori", "aug"], [ori_data, aug_data]):
        if case_data is None:
            continue  # Skip if no data available

        # Compute metrics grouped by drug and diagnosis
        grouped = case_data.groupby(groupby_col)
        sample_counts = case_data.groupby(groupby_col).size().reset_index(
            name='Sample count'
        )

        gt_col_name, pred_col_name = adjust_gt_col_name(case_data)

        for (group_keys, group) in grouped:
            if "model_name" in groupby_col:
                if "diagnosis" in groupby_col:
                    drug, diagnosis, model_name = group_keys
                    sample_count = sample_counts[
                        (sample_counts['diagnosis'] == diagnosis) & (sample_counts['Drug'] == drug)
                    ]['Sample count'].values[0]
                elif "Drug" in groupby_col:
                    drug, model_name = group_keys
                    diagnosis = "All"
                    sample_count = sample_counts[
                        sample_counts['Drug'] == drug
                    ]['Sample count'].values[0]
            else:
                if "diagnosis" in groupby_col:
                    drug, diagnosis = group_keys
                    model_name = "FNN"
                    sample_count = sample_counts[
                        (sample_counts['diagnosis'] == diagnosis) & (sample_counts['Drug'] == drug)
                        ]['Sample count'].values[0]
                elif "Drug" in groupby_col and len(groupby_col) == 1:
                    drug = group_keys
                    diagnosis = "All"
                    model_name = "FNN"
                    sample_count = sample_counts[
                        sample_counts['Drug'] == drug
                        ]['Sample count'].values[0]

            filtered_group = group.dropna(subset=[gt_col_name, pred_col_name])
            if len(filtered_group) < 2:  # Skip groups with insufficient data
                continue
            metrics = compute_metrics(filtered_group[gt_col_name], filtered_group[pred_col_name])

            ## Add clf metrics
            if if_with_clf_metrics:
                clf_metrics = compute_post_hoc_classification_metrics(filtered_group[gt_col_name], filtered_group[pred_col_name])
                metrics.update(clf_metrics)

            result_data.append({
                extract_label: extract_label_value,
                "ori_or_aug": case,
                    "model_name": model_name,
                "Drug": drug,
                "Drug (cancer count)": f"{drug} ({sample_count})",
                "TCGA Classification": diagnosis,
                "Diagnosis": diagnosis,
                "Sample count": sample_count,
                **metrics
            })
    if return_raw_data:
        raw_data_gt_pred = pd.concat([ori_data, aug_data])
        return result_data, raw_data_gt_pred
    else:
        return result_data


def load_saved_pred_gt_files(data_dir, file_pattern="No*.csv"):
    """
    Load prediction and ground truth data from the specified directory.

    Handles cases where ori and aug data are saved:
    - Individually in separate files.
    - Together in one file with an "ori_or_aug" column.

    Parameters:
        data_dir (str): Directory containing prediction files.
        file_pattern (str): Pattern to match prediction files.

    Returns:
        pd.DataFrame: DataFrame for ori data.
        pd.DataFrame: DataFrame for aug data.
    """
    files = glob(path.join(data_dir, file_pattern))
    ori_data = []
    aug_data = []

    for file in files:
        data = pd.read_csv(file)
        if "ori_or_aug" in data.columns:
            # Data already has ori_or_aug column
            ori_data.append(data[data["ori_or_aug"] == "ori"])
            aug_data.append(data[data["ori_or_aug"] == "aug"])
        else:
            # Assign ori_or_aug based on file name
            if 'ori' in file:
                data['ori_or_aug'] = 'ori'
                ori_data.append(data)
            elif 'aug' in file:
                data['ori_or_aug'] = 'aug'
                aug_data.append(data)

    ori_data_df = pd.concat(ori_data) if ori_data else None
    aug_data_df = pd.concat(aug_data) if aug_data else None
    return ori_data_df, aug_data_df

# def compute_metrics(ground_truth, predictions):
#     """
#     Compute performance metrics between ground truth and predictions.
#     Parameters:
#         ground_truth (pd.Series): Ground truth values.
#         predictions (pd.Series): Predicted values.
#     Returns:
#         dict: Dictionary containing the computed metrics.
#     """
#     spearman_corr, _ = spearmanr(ground_truth, predictions)
#     pearson_corr, _ = pearsonr(ground_truth, predictions)
#     r2 = r2_score(ground_truth, predictions)
#     rmse = mean_squared_error(ground_truth, predictions, squared=False)
#     return {
#             "Spearman Correlation": spearman_corr,
#             "Pearson Correlation": pearson_corr,
#             "R2 Score": r2,
#             "RMSE": rmse
#         }
#

def visualize_metrics_across_folders(file_pattern="No*.csv", extract_label="Num2aug"):
    """
    Visualize metrics across multiple directories.

    Parameters:
        data_dirs (list): List of directories to process.
        file_pattern (str): Pattern to match prediction files.
        extract_label (str): Label to extract hyperparameter from directory name.

    Returns:
        pd.DataFrame: Aggregated results.
    """
    data_dirs = [
            r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active projects\GexMix-ISMB\results\GDSC-RES\GDSC-aug-with-TCGA\12-26T20-18-TCGA_TCGA_noise0.05_num0",
            r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active projects\GexMix-ISMB\results\GDSC-RES\GDSC-aug-with-TCGA\12-26T20-18-TCGA_TCGA_noise0.05_num50",
            r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active projects\GexMix-ISMB\results\GDSC-RES\GDSC-aug-with-TCGA\12-26T20-18-TCGA_TCGA_noise0.05_num200",
            r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active projects\GexMix-ISMB\results\GDSC-RES\GDSC-aug-with-TCGA\12-26T20-18-TCGA_TCGA_noise0.05_num400",
    ]

    file_pattern = "*with_pred_gt_me*csv"

    for groupby_col in [["diagnosis", "Drug"], ["Drug"]]: ##
        result_data = []
        # groupby_col = ['diagnosis', 'Drug']
        groupby_col_str = "_".join(groupby_col)
        for data_dir in data_dirs:
            extract_label = "Num2aug"
            run_str = path.basename(data_dir)
            # Use a lambda function to extract the beta parameter
            param_extract_func = lambda data_dir: float(
                    path.basename(data_dir).split('_')[-1].split("num")[-1])

            result_data.extend(process_metrics_for_each_file_one_folder(data_dir, file_pattern,
                                                                        param_extract_func,
                                                                        groupby_col=['diagnosis', 'Drug'],
                                                                        extract_label=extract_label,
                                                                        if_with_clf_metrics=True,
                                                                        return_raw_data=False))
        # Aggregate results into a DataFrame
        results_df = pd.DataFrame(result_data).reset_index()

        # Compute mean and std for each group in results_df
        grouped_stats = results_df.groupby(["Drug", "ori_or_aug", extract_label]).agg(
            Spearman_Mean=("Spearman Correlation", "mean"),
            Spearman_Std=("Spearman Correlation", "std"),
            Pearson_Mean=("Pearson Correlation", "mean"),
            Pearson_Std=("Pearson Correlation", "std"),
            R2_Mean=("R2 Score", "mean"),
            R2_Std=("R2 Score", "std"),
            RMSE_Mean=("RMSE", "mean"),
            RMSE_Std=("RMSE", "std"),
        ).reset_index()

        ###Visualization (examples, modularize as needed)
        for metrics in ["F1-Score", "Accuracy", "AUC"]:
            for group_by_col in ["Diagnosis", "Drug"]:
                # Pivot the DataFrame so that Num2aug values appear as columns
                hyperparam_heatmap_group_by_col(
                        extract_label, results_df, prefix=f"{metrics}-({groupby_col_str})", metrics=metrics,
                        save_dir=data_dir, title_suffix=run_str, group_by=group_by_col)
            hyperparam_wise_boxplot(
                    results_df, extract_label=extract_label, prefix=f"{metrics}-({groupby_col_str})",
                    metrics=metrics, title_suffix=run_str,
                    save_dir=data_dir)
            # Categorize drugs by standard deviation for subplots
            hyperparam_effect_vis_categorized_by_std(
                    results_df, metrics=metrics, title_suffix=run_str,
                    prefix=f"{metrics}-({groupby_col_str})", extract_label=extract_label, save_dir=data_dir)
            # Set up subplots for each category
            hyperparam_effect_vis_mean_with_fill_between(
                    results_df, extract_label=extract_label, title_suffix=run_str,
                    metrics=metrics, prefix=f"{metrics}-({groupby_col_str})", save_dir=data_dir)
            hyperparam_effect_grouped_by_max_metric_group(
                    results_df, title_suffix=run_str,
                    extract_label=extract_label, metrics=metrics, prefix=f"{metrics}-({groupby_col_str})", save_dir=data_dir)
    return results_df


def visualize_metrics_across_folders_old():

    data_dirs = [
           # r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active "
           # r"projects\GexMix-ISMB\results\GDSC-RES\models\leave-cell-line-out\12-18T20"
           # r"-15_fnn_beta0.5_Rand_aug500",
           # r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active "
           # r"projects\GexMix-ISMB\results\GDSC-RES\models\leave-cell-line-out\12-18T20"
           # r"-15_fnn_beta1.0_Rand_aug500",
           # r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active "
           # r"projects\GexMix-ISMB\results\GDSC-RES\models\leave-cell-line-out\12-18T20"
           # r"-15_fnn_beta2.0_Rand_aug500",
           r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active "
           r"projects\GexMix-ISMB\results\GDSC-RES\models\leave-cell-line-out\12-18T19"
           r"-48_fnn_beta0.5_6zsco_aug100",
           r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active "
           r"projects\GexMix-ISMB\results\GDSC-RES\models\leave-cell-line-out\12-18T19"
           r"-48_fnn_beta1.0_6zsco_aug100",
           r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active "
           r"projects\GexMix-ISMB\results\GDSC-RES\models\leave-cell-line-out\12-18T19"
           r"-48_fnn_beta2.0_6zsco_aug100",
    ]

    file_pattern = "No*.csv"
    extract_label = "Num2aug"
    result_data = []
    num2aug = 50
    for jj, data_dir in enumerate(data_dirs):
        if jj == 0:
            ori_data, aug_data = load_saved_pred_gt_files(data_dir, file_pattern)
        else:
            _, aug_data = load_saved_pred_gt_files(data_dir, file_pattern)
        param = np.float32(path.basename(data_dir).split('_')[-3].split("beta")[-1])
        # param = np.int32(path.basename(data_dir).split('_')[-1].split("aug")[-1])
        for case, case_data in zip(["ori", "aug"], [ori_data, aug_data]):
            prefix = "DA with " + str(param) if case == "aug" else "No DA"
            # Compute metrics grouped by drug and cancer type
            grouped = case_data.groupby(['diagnosis', 'Drug'])
            # Compute the sample count for each group in grouped
            sample_counts = case_data.groupby(['diagnosis', 'Drug']).size().reset_index(
                    name='Sample count')
            for (diagnosis, drug), group in grouped:
                if len(group) < 2:  # Skip groups with insufficient data
                    continue
                print(
                        f"Processing: Diagnosis={diagnosis}, Drug={drug}, Sample count={group.shape[0]}")

                gt_col_name, pred_col_name = adjust_gt_col_name(case_data)

                metrics = compute_metrics(group['Ground Truth'], group['Predictions'])
                # Retrieve the sample count for the current diagnosis and drug
                sample_count = sample_counts[
                    (sample_counts['diagnosis'] == diagnosis) & (sample_counts['Drug'] == drug)
                    ]['Sample count'].values[0]  # Use `.values[0]` to extract the scalar value
                result_data.append(
                        {
                                extract_label: param,
                                "Case": prefix,
                                "Drug": drug,
                                "Drug (cancer count)": f"{drug} ({sample_count})",
                                "TCGA Classification": diagnosis,
                                "Diagnosis": diagnosis,
                                "Sample count": sample_count,
                                **metrics  # Unpack the dictionary directly
                        })
    results_df = pd.DataFrame(result_data).reset_index()
    # Compute mean and std for each group in results_df
    grouped_stats = results_df.groupby(["Drug", "Case", extract_label]).agg(
            Spearman_Mean=("Spearman Correlation", "mean"),
            Spearman_Std=("Spearman Correlation", "std"),
            Pearson_Mean=("Pearson Correlation", "mean"),
            Pearson_Std=("Pearson Correlation", "std"),
            R2_Mean=("R2 Score", "mean"),
            R2_Std=("R2 Score", "std"),
            RMSE_Mean=("RMSE", "mean"),
            RMSE_Std=("RMSE", "std")
    ).reset_index()

    # Filter the data for "No DA" case
    cat = "DA"
    if cat == "DA":
        da_data = results_df[~(results_df["Case"] == cat)]
        no_da_data = results_df[(results_df["Case"] == "No DA") & (results_df[extract_label] == 20)]
        no_da_data[extract_label] = 0
        need2plot_data = pd.concat([no_da_data, da_data], axis=0)
    elif cat == "NO DA":
        no_da_data = results_df[results_df["Case"] == cat]

    for metrics in ["Spearman Correlation", "Pearson Correlation"]:
        for group_by_col in ["Diagnosis", "Drug"]:
            # Pivot the DataFrame so that Num2aug values appear as columns
            hyperparam_heatmap_group_by_col(
                extract_label, need2plot_data, metrics=metrics,
                save_dir=data_dir, group_by=group_by_col)
        hyperparam_wise_boxplot(
            need2plot_data, extract_label=extract_label,
            metrics=metrics,
            save_dir=data_dir)
        # Categorize drugs by standard deviation for subplots
        hyperparam_effect_vis_categorized_by_std(
                need2plot_data, metrics=metrics, save_dir=data_dir)
        # Set up subplots for each category
        hyperparam_effect_vis_mean_with_fill_between(
                need2plot_data, extract_label=extract_label,
                metrics=metrics, save_dir=data_dir)
        hyperparam_effect_grouped_by_max_metric_group(
                need2plot_data,
                extract_label=extract_label, metrics=metrics, save_dir=data_dir)

def hyperparam_heatmap_group_by_col(extract_label, need2plot_data, metrics="Spearman Correlation",
                                    save_dir=None, prefix="", group_by="Cancer Type", title_suffix=""):
    # # Pivot the DataFrame so that Num2aug values appear as columns
    pivot_table2 = need2plot_data.pivot_table(
            index=group_by,
            columns=extract_label,
            values=metrics,
            aggfunc="mean"
    ).reset_index()
    # Ensure pivot_table2 is loaded and available
    # Perform hierarchical clustering on the rows (Cancer Types)
    row_linkage = linkage(pivot_table2.iloc[:, 1:], method='average', metric='euclidean')
    ordered_indices = leaves_list(row_linkage)
    # Reorder the pivot table based on clustering
    clustered_data = pivot_table2.iloc[ordered_indices, :]

    # Normalize only for colors (Min-Max normalization row-wise)
    normalized_data = clustered_data.iloc[:, 1:].apply(
        lambda row: (row - row.min()) / (row.max() - row.min()), axis=1)

    # Plot a clustered heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
            clustered_data.iloc[:, 1:],  # Exclude the index column
            annot=True,  # Annotate with actual values
            fmt=".2f",  # Format the values
            cmap="PiYG_r",  # Colormap for better visualization
            yticklabels=clustered_data[group_by],  # Use Cancer Type as y-axis labels,
    )
    # Add labels and title
    plt.title(f"Heatmap of {metrics} Grouped by {group_by}\n{title_suffix}")
    plt.xlabel(f"Parameter {extract_label}")
    plt.ylabel(group_by.capitalize())
    plt.tight_layout()
    plt.savefig(path.join(save_dir, f"0-heatmap_{prefix}_grouped_by_{group_by}.png"))
    plt.close()


def hyperparam_wise_boxplot(plot_data, extract_label="Num2aug", prefix="",
                            metrics="Spearman Correlation", if_with_p_value=False,
                            title_suffix="tit6le", save_dir="./"):
    ## only show hyparameter-wise distribution with pvalues
    # Pivot the DataFrame so that Num2aug values appear as columns
    pivot_table = plot_data.pivot_table(
            index="Drug",
            columns=extract_label,
            values=metrics,
            aggfunc="mean"
    ).reset_index()

    # Identify the Num2aug where each drug achieves its maximum Spearman Correlation
    max_num2aug_per_drug = pivot_table.set_index("Drug").iloc[:, 1:].idxmax(
            axis=1)  # Exclude the first column (Drug)
    # Add the max Num2aug to the pivot_table
    pivot_table[f"Max {extract_label}"] = max_num2aug_per_drug.values

    # Compute the mean Spearman Correlation for all drugs at each Num2aug
    pivot_table_means = pivot_table.set_index("Drug").iloc[:, :-1].mean().reset_index()
    pivot_table_means.columns = [extract_label, f"Mean {metrics}"]

    # Melt the pivot_table for plotting boxplots
    melted_data = pivot_table.melt(
            id_vars=["Drug", f"Max {extract_label}"],
            value_vars=pivot_table.columns[1:-1],  # Num2aug columns
            var_name=extract_label,
            value_name=metrics
    )

    # Convert Num2aug to numeric for proper sorting
    melted_data[extract_label] = melted_data[extract_label].astype(float)

    # Create the boxplot
    plt.figure(figsize=(14, 8))
    sns.boxplot(
            data=melted_data,
            x=extract_label,
            y=metrics,
            palette="tab20",
            showmeans=True,  # Show the mean marker on the boxplot
    )
    sns.stripplot(
            data=melted_data,
            x=extract_label,
            y=metrics,
            color="black",
            size=5,
            jitter=True,  # Add some jitter to separate overlapping points
            alpha=0.6
    )
    # Calculate the mean Spearman Correlation for each Num2aug
    grouped_means = melted_data.groupby(extract_label)[metrics]

    # Perform pairwise t-tests and annotate significance
    if if_with_p_value:
        # Get unique Num2aug values
        unique_num2aug = sorted(melted_data[extract_label].unique())

        # Perform pairwise t-tests and annotate significance
        for i in range(len(unique_num2aug) - 1):
            num2aug1 = unique_num2aug[i]
            num2aug2 = unique_num2aug[i + 1]

            # Get the distributions
            group1 = grouped_means.get_group(num2aug1)
            group2 = grouped_means.get_group(num2aug2)

            # Perform t-test
            t_stat, p_value = ttest_ind(group1, group2, equal_var=False)

            # Annotate significance level
            y_max = max(group1.max(), group2.max()) + 0.05
            x1, x2 = i, i + 1
            plt.plot([x1, x2], [y_max, y_max], color="black", lw=1.5)
            plt.text((x1 + x2) / 2, y_max + 0.02, f"p={p_value:.3f}", ha="center", color="red")

    # Add titles and labels
    plt.title(f"{metrics} Distribution Across {extract_label}\n{title_suffix}")
    plt.xlabel(f"Parameter {extract_label}")
    plt.ylabel(metrics)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path.join(save_dir, f"0-{prefix}-params-{extract_label}-boxplot.png"))
    plt.close()


def hyperparam_effect_grouped_by_max_metric_group(no_da_data, extract_label="Num2aug", prefix="",
                                                  metrics="Spearman Correlation", save_dir=None,
                                                  title_suffix=""):
    # Pivot the DataFrame so that Num2aug values appear as columns
    pivot_table = no_da_data.pivot_table(
            index="Drug",
            columns=extract_label,
            values=metrics,
            aggfunc="mean"
    ).reset_index()
    # Identify the Num2aug where each drug achieves its maximum Spearman Correlation
    max_num2aug_per_drug = pivot_table.set_index("Drug").iloc[:, 1:].idxmax(
        axis=1)  # Exclude the first column (Drug)
    # Add the max Num2aug to the pivot_table
    pivot_table[f"Max {extract_label}"] = max_num2aug_per_drug.values  # Align the values explicitly
    # Create subplots for drugs grouped by the Num2aug where they achieve their maximum performance
    unique_hyperparam = sorted(pivot_table[f"Max {extract_label}"].unique())
    # Create subplots for drugs grouped by the Num2aug where they achieve their maximum performance
    fig, axes = plt.subplots(
        len(unique_hyperparam), 1, figsize=(12, min(6 * len(unique_hyperparam), 11)), sharex=True)
    for i, hyperparam in enumerate(unique_hyperparam):
        ax = axes[i]
        subset = pivot_table[pivot_table[f"Max {extract_label}"] == hyperparam]

        for _, row in subset.iterrows():
            drug = row["Drug"]
            values = row.drop(["Drug", f"Max {extract_label}"]).values
            sns.lineplot(
                    x=subset.columns[1:-1].astype(float),  # Num2aug values as x-axis
                    y=values,  # Spearman Correlation values
                    ax=ax,
                    label=drug,
            )

        ax.set_title(f"Drugs with Max Performance at {extract_label}={hyperparam}\n{title_suffix}")
        ax.set_xlabel(f"Parameter {extract_label}")
        ax.set_ylabel(metrics)
        ax.legend(title="Drug", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path.join(save_dir, f"0-{prefix}-{extract_label}-drugs_max_sep.png"))
    plt.close()


def hyperparam_effect_vis_mean_with_fill_between(no_da_data, extract_label="Num2aug", prefix="",
                                                 metrics="Spearman Correlation", save_dir=None):
    # Generate unique drugs and corresponding dark colors
    uniq_drugs = no_da_data["Drug"].unique()
    colors = generate_dark_colors(len(uniq_drugs), dark_factor=0.65)
    plt.figure(figsize=[10, 7])
    for ind, dd in enumerate(uniq_drugs):
        drug_data = no_da_data[no_da_data["Drug"] == dd]
        if not drug_data.empty:
            drug_metric_mean = drug_data.groupby(extract_label)[metrics].mean()
            drug_metric_std = drug_data.groupby(extract_label)[metrics].std()
            sns.lineplot(
                    x=drug_metric_mean.index,
                    y=drug_metric_mean.values,
                    label=dd,
                    color=colors[ind],
                    marker="o"
            )
        plt.fill_between(
                drug_metric_mean.index,
                drug_metric_mean - drug_metric_std,
                drug_metric_mean + drug_metric_std,
                alpha=0.2,
                color=colors[ind]
        )
    plt.legend(
            title="Drug",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            ncol=1,
            frameon=False
    )
    plt.xlabel(extract_label)
    plt.ylabel(metrics)
    plt.tight_layout()
    plt.savefig(path.join(save_dir, f"0-{prefix}-{extract_label}-drugs_fill_between-{metrics}.png"))
    plt.close()

def hyperparam_effect_vis_categorized_by_std(no_da_data, extract_label="Num2aug", prefix="",
                                             metrics="Spearmanorrelation", save_dir=None,
                                             title_suffix=""):
    def categorize_std(std_values):
        """Categorize values into 4 percentile-based classes."""
        percentiles = [0, 25, 50, 75, 100]
        labels = [f"{percentiles[i]}th std." for i in range(len(percentiles) - 1)]
        return pd.qcut(std_values, q=[p / 100 for p in percentiles], labels=labels)

    # Initialize STD Category column
    no_da_data["STD Category"] = None
    # Compute and categorize standard deviations for each drug
    all_drug_stds = []
    drug_to_std = {}  # Map each drug to its categorized STD
    for dd in no_da_data["Drug"].unique():
        # Compute mean standard deviation for each drug
        drug_std_mean = no_da_data[no_da_data["Drug"] == dd].groupby(extract_label)[
            metrics].std().mean()
        all_drug_stds.append(drug_std_mean)
        drug_to_std[dd] = drug_std_mean
    # Categorize the computed mean STD values
    std_categories = categorize_std(all_drug_stds)
    # Assign STD Category to each drug in the dataset
    for dd, std_category in zip(drug_to_std.keys(), std_categories):
        no_da_data.loc[no_da_data["Drug"] == dd, "STD Category"] = std_category
    # Prepare to plot data in subplots by STD categories
    unique_std_categories = no_da_data["STD Category"].dropna().unique()
    fig, axes = plt.subplots(
            len(unique_std_categories), 1, figsize=(12, 6 * len(unique_std_categories)),
            sharex=True)
    # Generate unique colors for drugs
    uniq_drugs = no_da_data["Drug"].unique()
    colors = generate_dark_colors(len(uniq_drugs), dark_factor=0.65)
    # Plot each STD category in a separate subplot
    for ax, category in zip(axes, unique_std_categories):
        category_data = no_da_data[no_da_data["STD Category"] == category]
        for ind, dd in enumerate(uniq_drugs):
            drug_data = category_data[category_data["Drug"] == dd]
            if not drug_data.empty:
                drug_metric_mean = drug_data.groupby(extract_label)[metrics].mean()
                drug_metric_std = drug_data.groupby(extract_label)[metrics].std()

                sns.lineplot(
                        x=drug_metric_mean.index,
                        y=drug_metric_mean.values,
                        ax=ax,
                        label=dd,
                        color=colors[ind],
                        marker="o"
                )

                # Plot standard deviation as shaded regions
                ax.fill_between(
                        drug_metric_mean.index,
                        drug_metric_mean - drug_metric_std,
                        drug_metric_mean + drug_metric_std,
                        alpha=0.2,
                        color=colors[ind]
                )

        ax.set_title(f"Spearman Correlation - {category}\n{title_suffix}")
        ax.set_xlabel(extract_label)
        ax.set_ylabel(metrics)
        ax.legend(title="Drugs", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(path.join(save_dir, f"0-{prefix}-{extract_label}-drugs_sep_std-{metrics}.png"))
    plt.close()


# Function to generate and plot histograms of beta distributions
def plot_beta_histograms(alpha_values, num_bins=50, num_points=1000):
    alpha_values = [0.1, 0.5, 1, 2, 5]
    from scipy.stats import beta as Beta_distribution
    x = np.linspace(0.5, 1, num_points)  # Focus only on x > 0.5
    fig, axes = plt.subplots(1, len(alpha_values), figsize=(15, 3),)

    for i, alpha in enumerate(alpha_values):
        beta_param = alpha  # Use alpha=beta
        samples = Beta_distribution.rvs(alpha, beta_param, size=num_points)
        samples = samples[samples > 0.5]  # Focus on x > 0.5

        axes[i].hist(samples, bins=num_bins, density=True, color='royalblue', alpha=0.7)
        axes[i].set_title(f"alpha=beta={alpha}")
        axes[i].set_xlabel("Backbone sample mixing weight")

    axes[0].set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(path.join("../ISMB-paper", "beta-histogram.png"))
    plt.savefig(path.join("../ISMB-paper", "beta-histogram.pdf"), format="pdf")
    plt.close()


def load_pickle_file(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)

def extract_metrics(performance_metrics):
    ori_metrics = performance_metrics.get("ori")
    aug_metrics = performance_metrics.get("aug")
    return ori_metrics, aug_metrics

def convert_metrics_to_dataframe(ori_metrics, aug_metrics):
    coll_metric_ori_w_aug = []
    for metr_saved, name in zip([ori_metrics, aug_metrics], ["ori", "aug"]):
        dfs = []
        for metric, values in metr_saved.items():
            if "score" in metric:
                df = pd.DataFrame(values, columns=["Drug", "beta", metric])
                df["index"] = list(df.index)
                df["ori_or_aug"] = name
                dfs.append(df)
        result = dfs[0]
        for df in dfs[1:]:
            result = pd.merge(
                result, df, on=["index", "Drug", "beta", "ori_or_aug"], how="left")
        coll_metric_ori_w_aug.append(result)
    return pd.concat(coll_metric_ori_w_aug, axis=0)


def compute_CV_mean_summary_statistics(coll_metric_ori_w_aug,
                                       metrics_cols=['Spearman Correlation', 'Pearson Correlation', 'R2 Score'],
                                       groupby=['drug',  'ori_or_aug']):
    """
    Compute the mean and standard deviation of the performance metrics across cross-validation folds.
    :param coll_metric_ori_w_aug:
    :param groupby: columns = ['Drug', 'ori_or_aug', 'mean_spearman', 'std_spearman', 'mean_pearson',
       'std_pearson', 'mean_r2', 'std_r2']
    :return:
    """
    agg_dict = {}
    for metric in metrics_cols:
        metric_name = metric.split(' ')[0].lower()  # Normalize metric names for column naming
        if "mean" in metric:
            agg_dict[f'{metric_name}'] = (metric, 'mean')
            agg_dict[f'{metric_name}'] = (metric, 'std')
        else:
            agg_dict[f'mean_{metric_name}'] = (metric, 'mean')
            agg_dict[f'std_{metric_name}'] = (metric, 'std')

    return coll_metric_ori_w_aug.groupby(groupby).agg(**agg_dict).reset_index()


def compute_improvement(summary_stats, metrics=['spearman', 'pearson', 'r2']):
    """
    Compute the improvement of augmented metrics over original metrics.

    Args:
        summary_stats (pd.DataFrame): DataFrame with summarized statistics.
        metrics (list): List of metric names to compute improvements for.

    Returns:
        pd.DataFrame: DataFrame with computed improvements.
    """
    improvement = summary_stats.pivot_table(
            index='Drug', columns='ori_or_aug',
            values=[f'{metric}' for metric in metrics])
    improvement.columns = ['_'.join(col).strip() for col in improvement.columns.values]

    for metric in metrics:
        improvement[f'{metric}_improvement'] = improvement[f'{metric}_aug'] - improvement[f'{metric}_ori']

    improvement['Drug'] = improvement.index
    return improvement


def plot_heatmap(improvement, prefix="", sort_col="spearman_improvement", save_dir="./"):
    sort_imp = improvement.sort_values(sort_col, ascending=False)
    plt.figure(figsize=(8, 10))
    sns.heatmap(sort_imp[[sort_col]], annot=True, cmap="coolwarm", center=0)
    plt.title(f"Heatmap of Improvement (Aug - Ori)\n{prefix}")
    plt.xlabel("Improvement")
    plt.ylabel("Drug")
    plt.tight_layout()
    plt.savefig(path.join(save_dir, f"0-{prefix}-heatmap_improvement_{len(improvement)}drugs.png"))
    plt.close()


def plot_metric_heatmap(model_all_drugs, metric, x_axis="Drug", y_axis="Diagnosis", save_dir="./", prefix=""):
    """
    Plot a heatmap for a specific metric with drugs on the x-axis and diagnoses on the y-axis.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing the metric, drugs, and diagnoses.
    - metric (str): The column name of the metric to plot.
    - x_axis (str): The column name for the x-axis (default: "Drug").
    - y_axis (str): The column name for the y-axis (default: "Diagnosis").
    - save_path (str): File path to save the heatmap (optional).

    Returns:
    - None
    """
    # Pivot the data to create a matrix for the heatmap
    heatmap_data_ori = model_all_drugs[model_all_drugs["ori_or_aug"] == "ori"].pivot_table(
            index="Diagnosis", columns="Drug", values=metric, aggfunc="mean"
    )
    heatmap_data_aug = model_all_drugs[model_all_drugs["ori_or_aug"] == "aug"].pivot_table(
            index="Diagnosis", columns="Drug", values=metric, aggfunc="mean"
    )
    difference_heatmap_data = heatmap_data_aug - heatmap_data_ori

    difference_heatmap_data = difference_heatmap_data.apply(pd.to_numeric, errors='coerce')
    annot_data = difference_heatmap_data.copy()
    # annot_data = annot_data.applymap(lambda x: f"{x:.2f}" if (x != 0.0) & (~np.isnan(x)) else "")
    annot_data = annot_data.applymap(
            lambda x: f"{x:.2f}" if x != 0.0 and not np.isnan(x) else "0"
    )

    # Plot the heatmap
    plt.figure(figsize=(8, 7))
    sns.heatmap(
            difference_heatmap_data,
            annot=annot_data.values,
            cmap="coolwarm",
            linewidths=0.5,
            fmt='',
            center=0,
            cbar_kws={"label": metric},
    )
    plt.title(f"Heatmap of {metric} (Aug - Ori)\n{prefix}")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path.join(save_dir, f"0-{prefix}-heatmap-difference-{metric}.png"))
    plt.close()

def plot_facetgrid(plot_df, sort_improvement_df, prefix="",
                   top=True, selectK=5, plot_metrics=["Accuracy", "F1-Score"],
                   save_dir="./"):
    rename_hue = {"ori": "Original", "aug": "Augment"}
    plot_df["ori_or_aug"] = plot_df["ori_or_aug"].map(rename_hue)
    # Define custom color palette for renamed values
    custom_palette = {
            "Original": 'tab:blue',
            "Augment": 'tab:orange'
    }

    # Rename "ori_or_aug" values for better readability in the legend
    # plot_df = model_results_df.melt(
    #         id_vars=["Drug", "ori_or_aug", "Diagnosis", "model_name"],
    #         value_vars=plot_metrics,
    #         var_name="metric",
    #         value_name="score")  ## ['Drug', 'ori_or_aug', 'metric', 'score']
    # Select top or bottom improved drugs
    drugs = sort_improvement_df["Drug"].unique()[:selectK] if top else sort_improvement_df[
                                                                           "Drug"].unique()[
                                                                       -selectK:]
    subset = plot_df[plot_df["Drug"].isin(drugs)]

    # Create subplots
    fig, axes = plt.subplots(
        nrows=len(plot_metrics), ncols=1, figsize=(7, max(5, 3.8 * len(plot_metrics))), sharex=True)

    # Ensure axes is iterable even if there's only one subplot
    if len(plot_metrics) == 1:
        axes = [axes]
    unique_classes = plot_df["Drug"].unique()
    # Plot each metric
    for ax, metric in zip(axes, plot_metrics):
        for i, cls in enumerate(unique_classes):
            if i % 2 == 0:  # Add alternating gray background shades
                ax.axvspan(i - 0.5, i + 0.5, color='lightgray', alpha=0.3, zorder=0)
        # metric_data = drug_mean_std_metrics_long[drug_mean_std_metrics_long["metric"] == metric]
        sns.barplot(
                data=subset,
                x="Drug",
                y="mean_score",
                hue="ori_or_aug",
                palette=custom_palette,
                ax=ax,
                # ci=None,  # Disable Seaborn's internal confidence interval
                capsize=0.25,
        )
        ax.set_xlabel("Drug")
        ax.set_ylabel(metric)
        # ax.set_xticklabels(rotation=45, ha="right")
        ax.grid(alpha=0.3, axis="y")
        # Add legend only to the first subplot
        if ax == axes[0]:
            ax.legend(loc="upper left", borderaxespad=0.)
        else:
            ax.legend_.remove()
    plt.xticks(rotation=45, ha="right")
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the global title
    plt.savefig(path.join(save_dir, f"0-{prefix}-top{selectK}.png"))
    plt.close()

def plot_facetgrid_old(plot_df, sort_improvement_df, prefix="", top=True, selectK=5, save_dir="./"):
    # Rename "ori_or_aug" values for better readability in the legend
    rename_hue = {"ori": "Original", "aug": "Augment"}
    plot_df["ori_or_aug"] = plot_df["ori_or_aug"].map(rename_hue)

    drug_mean_std_metrics_long = plot_df.groupby(["Drug", "ori_or_aug", "metric", "model_name"]).agg(
            mean_score=("score", "mean"),
            std_score=("score", "std")
    ).reset_index()

    # Define custom color palette for renamed values
    custom_palette = {
            "Original": 'tab:blue',
            "Augment": 'tab:orange'
    }

    # Select top or bottom improved drugs
    drugs = sort_improvement_df["Drug"].unique()[:selectK] if top else sort_improvement_df[
                                                                           "Drug"].unique()[
                                                                       -selectK:]
    subset = drug_mean_std_metrics_long[drug_mean_std_metrics_long["Drug"].isin(drugs)]

    title = f"Top {selectK} improved drugs" if top else f"Bottom {selectK} worsened drugs"

    # Create the catplot
    g = sns.catplot(
            data=subset,
            x="Drug", y="mean_score", hue="ori_or_aug", col="metric", palette=custom_palette,  # Use the custom palette
            kind="bar", ci="sd", col_wrap=2, height=3, aspect=1.5, sharex=False, sharey=False
    )
    # g.set_xticklabels(["Men", "Women", "Children"])
    # Rotate x-tick labels for all subplots
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(alpha=0.3)

    # Remove the FacetGrid legend (if not needed)
    if g._legend:
        g._legend.remove()

    # Add a custom legend (optional)
    plt.legend(title="Type", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)

    # Set subplot titles and adjust layout
    g.set_titles(title + " {col_name}")
    plt.tight_layout()

    plt.savefig(path.join(save_dir, f"0-{prefix}-{title}-select{selectK}.png"))
    plt.close()


def compute_log_fold_change_and_paired_t_tests(summary_stats, metrics_cols=['mean_spearman', 'mean_pearson', 'mean_r2']):
    from scipy.stats import ttest_rel
    pivoted = summary_stats.pivot_table(
            index='Drug', columns='ori_or_aug',
            values=metrics_cols
    )
    pivoted.columns = ['_'.join(col).strip() for col in pivoted.columns.values]
    pivoted = pivoted.reset_index()
    for metric in metrics_cols:
        pivoted[f'logfc_{metric}'] = np.log2(
                pivoted[f'{metric}_aug'] / pivoted[f'{metric}_ori']
        )
    p_values = {}
    for metric in metrics_cols:
        ori = pivoted[f'{metric}_ori']
        aug = pivoted[f'{metric}_aug']
        t_stat, p_val = ttest_rel(ori, aug)
        p_values[f'p_value_{metric}'] = -np.log10(p_val)
    for metric in metrics_cols:
        pivoted[f'p_value_{metric}'] = p_values[f'p_value_{metric}']
    return pivoted


def plot_slope_chart(summary_stats, column="ori_or_aug", prefix="case_name", value2plot="mean_spearman", save_dir="./"):
    pivot_df = summary_stats.pivot(
            index='Drug', columns=column, values=value2plot).reset_index()
    plt.figure(figsize=(8, 6))
    for _, row in pivot_df.iterrows():
        plt.plot(['ori', 'aug'], [row['ori'], row['aug']], color='gray', alpha=0.5)
        if row['aug'] > row['ori']:
            plt.plot(
                    ['ori', 'aug'], [row['ori'], row['aug']], color='green', linewidth=2,
                    alpha=0.5)
        else:
            plt.plot(
                    ['ori', 'aug'], [row['ori'], row['aug']], color='red', linewidth=2,
                    alpha=0.5)
    plt.scatter(['ori'] * len(pivot_df), pivot_df['ori'], color='blue', label='Ori', zorder=3)
    plt.scatter(['aug'] * len(pivot_df), pivot_df['aug'], color='orange', label='Aug', zorder=3)
    plt.title(f"Slope Chart: {value2plot.capitalize()}\n{prefix}", fontsize=14)
    plt.ylabel(f"{value2plot.capitalize()}", fontsize=12)
    plt.xticks(['ori', 'aug'], labels=['Original (Ori)', 'Augmented (Aug)'], fontsize=12)
    plt.legend(title="Data Type")
    plt.tight_layout()
    plt.savefig(path.join(save_dir, f"0-{prefix}-slope_chart_{value2plot}.png"))
    plt.close()

"""
lookup_drugs = ["Nilotinib", "Tanespimycin", "PHA-665752", "Lapatinib", "Nutlin-3a", "Saracatinib", 
"Crizotinib", 
"Panobinostat", "Sorafenib", "Irinotecan", "Topotecan", "PD0325901", "Palbociclib", "Paclitaxel", 
"Selumetinib", "PLX-4720", "NVP-TAE684", "Erlotinib"]

"""

### plot metrics of all drugs in one folder
def load_saved_csvs_get_stats_of_one_folder(ori_data_dir, aug_data_dir, file_patten="No*csv", if_use_clf_metrics=False):
    # ori_file_dir = (r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active projects\GexMix-ISMB\results\GDSC-RES\models\leave-cell-line-out\12-16T23-38_fnn_beta0.5_6zsco_aug100")
    ori_files = glob(path.join(ori_data_dir, file_patten))

    # aug_file_dir = r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active projects\GexMix-ISMB\results\GDSC-RES\models\leave-cell-line-out\12-16T23-38_fnn_beta1.0_6zsco_aug100"
    aug_files = glob(path.join(aug_data_dir, file_patten))

    coll_processed_data = []
    for case_data, ori_or_aug in zip([ori_files, aug_files], ["ori", "aug"]):
        case_processed_data = []
        if len(case_data) > 0:
            for file in ori_files:
                data = pd.read_csv(file)
                grouped = data.groupby(['diagnosis', 'Drug'])
                # Compute the sample count for each group in grouped
                sample_counts = data.groupby(['diagnosis', 'Drug']).size().reset_index(
                        name='Sample count')
                for (diagnosis, drug), group in grouped:
                    print(
                            f"Processing: Diagnosis={diagnosis}, Drug={drug}, Sample count={group.shape[0]}")
                    filted_group = group.dropna(subset=['Ground Truth', 'Predictions'])
                    if len(filted_group) < 2:
                        continue
                    metrics = compute_metrics(filted_group['Ground Truth'], filted_group['Predictions'])
                    # Retrieve the sample count for the current diagnosis and drug
                    sample_count = sample_counts[
                        (sample_counts['diagnosis'] == diagnosis) & (sample_counts['Drug'] == drug)
                        ]['Sample count'].values[0]  # Use `.values[0]` to extract the scalar value
                    case_processed_data.append(
                            {
                                    "Drug": drug,
                                    "ori_or_aug": ori_or_aug,
                                    "Drug (cancer count)": f"{drug} ({sample_count})",
                                    "TCGA Classification": diagnosis,
                                    "Diagnosis": diagnosis,
                                    "Sample count": sample_count,
                                    **metrics  # Unpack the dictionary directly
                            })
            case_processed_data_df = pd.DataFrame(case_processed_data).reset_index()
            coll_processed_data.append(case_processed_data_df)
    coll_processed_data_df = pd.concat(coll_processed_data, axis=0).reset_index()
    return coll_processed_data_df


def load_saved_pickle_get_stats_of_one_folder(ori_data_dir, aug_data_dir=None):

    if not aug_data_dir:
        aug_data_dir = ori_data_dir
        performance_metrics = load_pickle_file(ori_data_dir)
        ori_metrics, aug_metrics = extract_metrics(performance_metrics)
        coll_metric_ori_w_aug = convert_metrics_to_dataframe(ori_metrics, aug_metrics)
    else:
        performance_metrics_ori = load_pickle_file(ori_data_dir)
        performance_metrics_aug = load_pickle_file(aug_data_dir)
        ori_metrics, _ = extract_metrics(performance_metrics_ori)
        _, aug_metrics = extract_metrics(performance_metrics_aug)
        coll_metric_ori_w_aug = convert_metrics_to_dataframe(ori_metrics, aug_metrics)
    return coll_metric_ori_w_aug

def generate_latex_table_with_mean_std_predefined_order(
    result_df, metric_columns, groupby_column, table_caption, table_label, metrics_order, only_row_latex=False
):
    """
    Generate a LaTeX table dynamically from a DataFrame using mean ± std format for metrics,
    with a predefined order of metrics.
    Parameters:
        result_df (pd.DataFrame): DataFrame containing the results.
        metric_columns (list): List of metric column names to include in the table.
        groupby_column (str): Column to group by for table rows.
        table_caption (str): Caption for the LaTeX table.
        table_label (str): Label for referencing the table in LaTeX.
        metrics_order (list): Predefined order of metrics for the table.
    Returns:
        str: LaTeX code for the table.
    """
    # Prepare LaTeX table header
    header = r"""
    \begin{table}[t]
    \centering
    \caption{%s\label{%s}}
    \tabcolsep=0pt%%
    \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l%s@{\extracolsep{\fill}}}
    \toprule
    & \multicolumn{2}{c}{Without DA (Ori)} & \multicolumn{2}{c}{With DA (Aug Case)} \\
    \cline{2-3}\cline{4-5}
    Models & %s \\
    \midrule
    """ % (
            table_caption,
            table_label,
            "c" * (2 * len(metrics_order)),
            " & ".join([f"${metric}$" for metric in metrics_order * 2]),
    )
    # Calculate descriptive statistics
    grouped = result_df.groupby([groupby_column, 'ori_or_aug'])[metric_columns].describe()
    # Prepare rows for each group
    rows = []
    for model_name in result_df[groupby_column].unique():
        ori_stats = grouped.loc[(model_name, 'ori')]
        aug_stats = grouped.loc[(model_name, 'aug')]
        # Handle pd.Series for single groups
        if isinstance(ori_stats, pd.Series):
            ori_metrics_str = " & ".join(
                    [
                            f"{ori_stats[col, 'mean']:.2f} $\\pm$ {ori_stats[col, 'std']:.2f}"
                            for col in metrics_order
                    ]
            )
        else:
            ori_metrics_str = " & ".join(
                    [
                            f"{ori_stats.loc[col]['mean']:.2f} $\\pm$ {ori_stats.loc[col]['std']:.2f}"
                            for col in metrics_order
                    ]
            )
        if isinstance(aug_stats, pd.Series):
            aug_metrics_str = " & ".join(
                    [
                            f"{aug_stats[col, 'mean']:.2f} $\\pm$ {aug_stats[col, 'std']:.2f}"
                            for col in metrics_order
                    ]
            )
        else:
            aug_metrics_str = " & ".join(
                    [
                            f"{aug_stats.loc[col]['mean']:.2f} $\\pm$ {aug_stats.loc[col]['std']:.2f}"
                            for col in metrics_order
                    ]
            )
        rows.append(f"{model_name} & {ori_metrics_str} & {aug_metrics_str} \\\\")
    # Combine all rows
    body = "\n".join(rows)
    # Prepare LaTeX table footer
    footer = r"""
    \botrule
    \end{tabular*}
    \end{table}
    """
    # Combine all parts
    if only_row_latex:
        latex_table = body
    else:
        latex_table = header + body + footer
    print(latex_table)
    return latex_table


def load_analyze_plot_metrics_of_one_folder(file_fomat="pickle"):
    """
    This is overarching function
    for getting the metrics of all drugs in one folder from either the
    saved pickle files (load_saved_pickle_get_stats_of_one_folder)or the saved csv files
    load_saved_csv_get_stats_of_one_folder.
    :return:
    """

    data_dirs = [
            r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active projects\GexMix-ISMB\results\GDSC-RES\models\leave-cell-line-out\12-29T12-14_multiModels_beta1.0_Rand_aug500",
            # r"..\results\GDSC-RES\GDSC-aug-with-TCGA\12-20T23-24-GDSC_TCGA_beta0.6_num50",

    ]
    result_data = []
    file_pattern = "*with_pred_gt_me*csv"
    for data_dir in data_dirs:
        folder_str = path.basename(data_dir)
        extract_label = "Num2aug"
        # Use a lambda function to extract the beta parameter
        param_extract_func = lambda data_dir: float(
                path.basename(data_dir).split('_')[-1].split("num")[-1])

        folder_results_list = process_metrics_for_each_file_one_folder(
            data_dir, file_pattern,
            param_extract_func, groupby_col=["Drug", "diagnosis", "model_name"],
            extract_label=extract_label, return_raw_data=False,
            if_with_clf_metrics=True)
        folder_results_df = pd.DataFrame(folder_results_list)
        folder_results_df.to_csv(path.join(data_dir, f"{folder_str}_drug_wise_all_metrics.csv"), index=False)

        ## generate latex table of high level metrics
        generate_latex_table_with_mean_std_predefined_order(
                folder_results_df, metric_columns=["Spearman Correlation", "Pearson Correlation", "Accuracy", "F1-Score"],
                groupby_column="model_name",
                table_caption="Summary of performance metrics with and without data augmentation.",
                table_label="tab:performance_metrics",
                metrics_order=["Spearman Correlation", "Pearson Correlation", "Accuracy", "F1-Score"]
        )
        """Get mean over all samples of one drug
        ['Drug', 'ori_or_aug', 'mean_spearman', 'std_spearman',  'mean_pearson',
           'std_pearson', 'mean_r2', 'std_r2']"""
        drug_cv_metric_mean_stats = compute_CV_mean_summary_statistics(
            folder_results_df,
                metrics_cols=['Spearman Correlation', 'Pearson Correlation', 'Accuracy', 'F1-Score'],
                groupby=["Drug",  "ori_or_aug", "Diagnosis", "model_name"])

        uniq_models = drug_cv_metric_mean_stats["model_name"].unique()
        for model_name in ["FNN"]:
            model_all_drugs = drug_cv_metric_mean_stats[
                drug_cv_metric_mean_stats["model_name"] == model_name]
            drug_mean_metric_improvement = compute_improvement(
                    model_all_drugs,
                    metrics=['mean_spearman', 'mean_pearson', "mean_accuracy", "mean_f1-score"])
            # Melt the dataframe for long-form representation for plotting
            plot_df = model_all_drugs.melt(
                    id_vars=["Drug", "ori_or_aug", "Diagnosis", "model_name"],
                    value_vars=["mean_pearson", "mean_spearman", "mean_accuracy", "mean_f1-score"],
                    var_name="metric", value_name="score")
            drug_mean_std_metrics_long = plot_df.groupby(
                    ["Drug", "ori_or_aug", "Diagnosis", "model_name", "metric"]).agg(
                    mean_score=("score", "mean"),
                    std_score=("score", "std")
            ).reset_index()
            plot_heatmap(
                    drug_mean_metric_improvement, sort_col="mean_spearman_improvement",
                    prefix=folder_str, save_dir=data_dir)

            plot_metric_heatmap(
                model_all_drugs, "mean_spearman", x_axis="Drug", y_axis="Diagnosis",
                    save_dir=data_dir, prefix=folder_str)

            plot_facetgrid(
                    drug_mean_std_metrics_long, drug_mean_metric_improvement.sort_values(
                            "mean_spearman_improvement", ascending=False),
                    top=True, prefix=folder_str + f"{model_name}", selectK=20,
                    plot_metrics=["mean_spearman", "mean_f1-score"],
                    save_dir=data_dir)

            model_drug_cv_metric_mean_stats = compute_CV_mean_summary_statistics(
                    model_all_drugs,
                    metrics_cols=["mean_pearson", "mean_spearman", "mean_accuracy",
                                  "mean_f1-score"],
                    groupby=["Drug", "ori_or_aug", "model_name"])
            drug_mean_logfc_p_df = compute_log_fold_change_and_paired_t_tests(
                    model_drug_cv_metric_mean_stats,
                    metrics_cols=['mean_spearman', 'mean_pearson',
                                  "mean_accuracy", "mean_f1-score"])
            plot_slope_chart(
                    model_drug_cv_metric_mean_stats, column="ori_or_aug", prefix=folder_str,
                    value2plot="mean_spearman", save_dir=data_dir)
            ## melt results_df for plotting
            melt_results_df = model_all_drugs.melt(
                    id_vars=["Drug", "ori_or_aug", "Diagnosis"],
                    value_vars=["mean_pearson", "mean_spearman", "mean_accuracy",
                                "mean_f1-score"],
                    var_name="metric", value_name="score")
            plot_boxstripplot_ori_and_aug_overall_drugs(
                    melt_results_df, metric="mean_pearson", prefix=folder_str, y_col="score",
                    save_dir=data_dir)
            compute_p_values_paired_metrics(
                model_drug_cv_metric_mean_stats, metrics_cols=['mean_spearman', 'mean_pearson'])

def plot_boxstripplot_ori_and_aug_overall_drugs(plot_df, metric="mean_pearson", y_col="score",
                                                prefix="", save_dir="./"):
    plt.figure(figsize=(8, 6))
    plot_df = plot_df[plot_df["metric"] == metric]
    sns.boxplot(data=plot_df, x='ori_or_aug', y=y_col, palette='Set2', width=0.5)
    sns.stripplot(
        data=plot_df, x='ori_or_aug', y=y_col, palette='Set2', size=6, jitter=True, alpha=0.7)

    t_stat, p_value = ttest_ind(
            plot_df[plot_df["ori_or_aug"] == "ori"][y_col],
            plot_df[plot_df["ori_or_aug"] == "aug"][y_col])

    if p_value <= 0.05:
        if p_value < 0.001:
            sig_str = "***"
        elif p_value < 0.01:
            sig_str = "**"
        else:
            sig_str = "*"
        # Annotate the p-value
        x1, x2 = 0, 1  # Position of 'ori' and 'aug' on the x-axis
        y_max = plot_df[y_col].max() + 0.1  # Height for the annotation
        plt.plot([x1, x2], [y_max, y_max], color='black', linewidth=1.5)  # Line connecting groups
        plt.text(
                (x1 + x2) / 2, y_max + 0.5, f'{sig_str}', ha='center', va='bottom', fontsize=12)

    # Customize the plot
    plt.title(f"Comparison of Ori and Aug Drugs\n{prefix}", fontsize=14)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.xlabel("Condition", fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(path.join(save_dir, f"0-{prefix}-{metric}_ori_vs_aug_stripboxplot.png"))
    plt.close()



def compute_p_values_paired_metrics(drug_cv_metric_mean_stats,
                                    metrics_cols=['mean_spearman', 'mean_pearson', 'mean_r2']):
    pivot_paired_metrics_df = drug_cv_metric_mean_stats.pivot_table(
            index='Drug', columns='ori_or_aug', values=metrics_cols)

    for test_func in [ttest_rel, wilcoxon]:
        for metric in metrics_cols:
            ori = pivot_paired_metrics_df[metric]['ori']
            aug = pivot_paired_metrics_df[metric]['aug']
            t_stat, p_val = test_func(ori, aug)
            print(f"{test_func.__name__} {metric}: t-stat={t_stat:.3f}, p-value={p_val:.3e}")

    # # Perform paired t-tests
    # t_stat_spearman, p_val_spearman = ttest_rel(
    #         pivot_paired_metrics_df['mean_spearman']['aug'],
    #         pivot_paired_metrics_df['mean_spearman']['ori'])
    # t_stat_pearson, p_val_pearson = ttest_rel(
    #         pivot_paired_metrics_df['mean_pearson']['aug'],
    #         pivot_paired_metrics_df['mean_pearson']['ori'])
    # t_stat_r2, p_val_r2 = ttest_rel(
    #         pivot_paired_metrics_df['mean_r2']['aug'], pivot_paired_metrics_df['mean_r2']['ori'])
    # print(f"Spearman Score: t-stat={t_stat_spearman:.3f}, p-value={p_val_spearman:.3e}")
    # print(f"Pearson Score: t-stat={t_stat_pearson:.3f}, p-value={p_val_pearson:.3e}")
    # print(f"R2 Score: t-stat={t_stat_r2:.3f}, p-value={p_val_r2:.3e}")
    # wilcoxon_spearman = wilcoxon(
    #         pivot_paired_metrics_df['mean_spearman']['aug'],
    #         pivot_paired_metrics_df['mean_spearman']['ori'])
    # wilcoxon_pearson = wilcoxon(
    #         pivot_paired_metrics_df['mean_pearson']['aug'],
    #         pivot_paired_metrics_df['mean_pearson']['ori'])
    # wilcoxon_r2 = wilcoxon(
    #         pivot_paired_metrics_df['mean_r2']['aug'], pivot_paired_metrics_df['mean_r2']['ori'])
    # print(f"Spearman Score Wilcoxon: p-value={wilcoxon_spearman.pvalue:.3e}")
    # print(f"Pearson Score Wilcoxon: p-value={wilcoxon_pearson.pvalue:.3e}")
    # print(f"R2 Score Wilcoxon: p-value={wilcoxon_r2.pvalue:.3e}")


## generate pickle dataset with shared genes between L1000 and TCGA/CCLE
def generate_pickle_with_L1000_genes():
    L1000_df = pd.read_csv(
        r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active projects\GexMix-ISMB\data\GSE92742_Broad_LINCS_gene_info.txt",
        sep="\t")
    L1000_genes = L1000_df[L1000_df["pr_is_lm"] == 1]["pr_gene_symbol"]
    with open(
            r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active projects\GexMix-ISMB\data\tcga_10374_5000_std.pickle",
            "rb") as file:
        related_labels_dict = pickle.load(file)
        tcga_meta_data = related_labels_dict["meta"]
    tcga_data = pd.read_csv(
        r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active projects\1-pancan_subtyping\data\1_filter_CCLE_with_TCGA\filtered_TCGA_with_shared_gene_between_CCLE_10374_17713.csv")
    tcga_data.index = tcga_data["Unnamed: 0"]
    shared_genes_tcga = set(tcga_data.columns).intersection(set(L1000_genes))
    combined_norm_data = {}
    combined_norm_data["rnaseq"] = tcga_data[list(shared_genes_tcga)]
    combined_norm_data["meta"] = tcga_meta_data
    with open(
            f"../data/tcga_{len(tcga_data)}_{len(shared_genes_tcga)}_L1000.pickle", "wb") as handle:
        pickle.dump(combined_norm_data, handle)

    ### CCLE
    with open(
            r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active projects\GexMix-ISMB\data\ccle_1248_5000_std.pickle",
            "rb") as file:
        related_labels_dict = pickle.load(file)
        ccle_meta_data = related_labels_dict["meta"]
        ccle_gex_data = related_labels_dict["rnaseq"]

    ccle_data = pd.read_csv(
        r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active projects\1-pancan_subtyping\data\1_filter_CCLE_with_TCGA\filtered_CCLE_with_shared_gene_between_TCGA_1248_17713_renameCol.csv")
    shared_genes_ccle = set(ccle_data.columns).intersection(set(L1000_genes))
    ccle_norm_data = {}
    ccle_norm_data["rnaseq"] = ccle_data[list(shared_genes_tcga)]
    ccle_norm_data["meta"] = ccle_meta_data
    with open(
            f"../data/ccle_{len(ccle_data)}_{len(shared_genes_tcga)}_L1000.pickle",
            "wb") as handle:
        pickle.dump(ccle_norm_data, handle)

    ### GDSC
    with open(
            r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active projects\GexMix-ISMB\data\GDSC_drug_response\GDSC_gex_w_filtered_meta_(959, 19434)_processed9.pickle",
            "rb") as file:
        related_labels_dict = pickle.load(file)
        gdsc_meta_data = related_labels_dict["meta"]
        gdsc_gex_data = related_labels_dict["rnaseq"]
    shared_genes_gdsc = set(gdsc_gex_data.columns).intersection(set(L1000_genes))

    shared_genes_ccle_gdsc = list(shared_genes_ccle.intersection(shared_genes_gdsc))
    gdsc_norm_data = {}
    gdsc_norm_data["rnaseq"] = ccle_data[list(shared_genes_ccle_gdsc)]
    gdsc_norm_data["meta"] = ccle_meta_data
    with open(
            f"../data/ccle_{len(ccle_data)}_{len(shared_genes_ccle_gdsc)}_L1000.pickle",
            "wb") as handle:
        pickle.dump(ccle_norm_data, handle)



def compute_post_hoc_classification_metrics(ground_truth, predictions,
                                            thresholds=[0.25, 0.75],
                                            class_mapping={0: 0, 1: 0.5, 2: 1}):
    """
    Evaluate post hoc classification performance metrics for continuous predictions.

    Args:
        predictions (np.array): Continuous predictions.
        ground_truth (np.array): Discrete ground truth values (e.g., 0, 0.5, 1).
        thresholds (list): Thresholds for mapping continuous predictions to discrete classes.

    Returns:
        dict: Dictionary of performance metrics.
    """
    # Map predictions to discrete classes using thresholds
    discrete_predictions = np.digitize(predictions, bins=thresholds)
    # class_mapping = {0: 0, 1: 0.5, 2: 1}  # Map indices to discrete classess
    discrete_predictions2 = np.vectorize(class_mapping.get)(discrete_predictions)

    discrete_ground_truth = np.digitize(ground_truth, bins=thresholds)
    discrete_ground_truth2 = np.vectorize(class_mapping.get)(discrete_ground_truth)
    ### Compute classification metrics
    accuracy = np.mean(discrete_predictions == discrete_ground_truth)

    precision = precision_score(
            discrete_ground_truth, discrete_predictions, average='weighted', zero_division=0)
    recall = recall_score(
        discrete_ground_truth, discrete_predictions, average='weighted', zero_division=0)
    f1 = f1_score(discrete_ground_truth, discrete_predictions, average='weighted', zero_division=0)
    confusion = confusion_matrix(discrete_ground_truth, discrete_predictions)

    # Compute AUC for each class and overall weighted AUC
    try:
        auc_per_class = roc_auc_score(
                y_true=np.eye(3)[discrete_ground_truth],  # One-hot encoding of ground truth
                y_score=np.eye(3)[discrete_predictions],  # One-hot encoding of predictions
                multi_class='ovr',  # One-vs-Rest AUC calculation
                average=None  # Compute AUC for each class
        )
        weighted_auc = roc_auc_score(
                y_true=np.eye(3)[discrete_ground_truth],
                y_score=np.eye(3)[discrete_predictions],
                multi_class='ovr',
                average='weighted'  # Weighted average AUC
        )
    except ValueError:
        auc_per_class = [None] * 3  # Placeholder if AUC cannot be computed
        weighted_auc = None

    #### Compute classification metrics Compute regression-like metrics (MSE, MAE)
    mse = np.mean((predictions - ground_truth) ** 2)
    mae = np.mean(np.abs(predictions - ground_truth))
    # Combine all metrics
    metrics = {
            "Accuracy": accuracy,
            "AUC": weighted_auc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Confusion Matrix": confusion.tolist(),
            "MSE": mse,
            "MAE": mae
    }

    return metrics


def compute_grouped_mean_std_of_metrics_generate_latex(coll_processed_data_df, groupby=['Drug', 'ori_or_aug']):
    """
    Compute the mean and standard deviation of the performance metrics across cross-validation folds.
    :param coll_processed_data_df: computed drug-cancer wise metrics df
    :return:
    """

    def generate_latex_row_with_pm(metrics_summary, model_name):
        # Filter metrics for the given model
        ori_metrics = metrics_summary[metrics_summary['ori_or_aug'] == 'ori']
        aug_metrics = metrics_summary[metrics_summary['ori_or_aug'] == 'aug']
        if ori_metrics.empty or aug_metrics.empty:
            return f"% No data available for {model_name}\n"

        # Compare and format values
        def bold_if_greater(ori_value, aug_value):
            ori_str = f"{ori_value:.2f}"
            aug_str = f"{aug_value:.2f}"
            if aug_value > ori_value:
                return f"${ori_str}", f"$\\textbf{{{aug_str}}}"
            else:
                return f"$\\textbf{{{ori_str}}}", f"${aug_str}"

        ori_pearson, aug_pearson = bold_if_greater(
                ori_metrics['Pearson_Mean'].values[0], aug_metrics['Pearson_Mean'].values[0])
        ori_spearman, aug_spearman = bold_if_greater(
                ori_metrics['Spearman_Mean'].values[0], aug_metrics['Spearman_Mean'].values[0])
        ori_f1, aug_f1 = bold_if_greater(
                ori_metrics['F1_Mean'].values[0], aug_metrics['F1_Mean'].values[0])
        ori_accuracy, aug_accuracy = bold_if_greater(
                ori_metrics['Accuracy_Mean'].values[0], aug_metrics['Accuracy_Mean'].values[0])
        # Add std values with $\pm$
        ori_pearson += f" \\pm {ori_metrics['Pearson_Std'].values[0]:.2f}$"
        aug_pearson += f" \\pm {aug_metrics['Pearson_Std'].values[0]:.2f}$"
        ori_spearman += f" \\pm {ori_metrics['Spearman_Std'].values[0]:.2f}$"
        aug_spearman += f" \\pm {aug_metrics['Spearman_Std'].values[0]:.2f}$"
        ori_f1 += f" \\pm {ori_metrics['F1_Std'].values[0]:.2f}$"
        aug_f1 += f" \\pm {aug_metrics['F1_Std'].values[0]:.2f}$"
        ori_accuracy += f" \\pm {ori_metrics['Accuracy_Std'].values[0]:.2f}$"
        aug_accuracy += f" \\pm {aug_metrics['Accuracy_Std'].values[0]:.2f}$"
        # Format the LaTeX row
        latex_row = f"{model_name} & {ori_pearson} & {ori_spearman} & {ori_accuracy} &{ori_f1} &  {aug_pearson} & {aug_spearman} & {aug_accuracy}& {aug_f1}  \\\\"
        # Display the generated LaTeX row
        print(latex_row)
        return latex_row

    metrics_summary = coll_processed_data_df.groupby(groupby).agg(
            Pearson_Mean=('Pearson Correlation', 'mean'),
            Pearson_Std=('Pearson Correlation', 'std'),
            Spearman_Mean=('Spearman Correlation', 'mean'),
            Spearman_Std=('Spearman Correlation', 'std'),
            F1_Mean=('F1-Score', 'mean'),
            F1_Std=('F1-Score', 'std'),
            Accuracy_Mean=('Accuracy', 'mean'),
            Accuracy_Std=('Accuracy', 'std'),
    ).reset_index()
    # Example usage for "Linear regression"
    model_name = "Linear regression"
    latex_row_pm = generate_latex_row_with_pm(metrics_summary, model_name)


def load_TCGA_saved_multi_models_get_vis():
    coll_processed_data = []
    data_dirs = [
            r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active projects\GexMix-ISMB\results\GDSC-RES\GDSC-aug-with-TCGA\12-20T23-24-TCGA_TCGA_beta2_num200"
    ]
    for data_dir in data_dirs:
        csv_files = glob(path.join(data_dir, "*pred_gt_meta*csv"))
        trial_name = path.basename(data_dir).split("-")[-1]
        for file in csv_files:
            data = pd.read_csv(file)
            beta = path.basename(data_dir).split("_")[-2].split("beta")[-1]
            num2aug = path.basename(data_dir).split("_")[-1].split("num")[-1]
            data["model_name"] = data["model_name"].astype(str)
            # Separate the first 50 rows
            first_50_rows = data.iloc[:len(data) // 2]
            # Identify rows where "ori_or_aug" == 'aug' and "model_name" is not NaN
            aug_model_names = first_50_rows.loc[first_50_rows['ori_or_aug'] == 'aug', 'model_name']
            data.loc[first_50_rows.index, 'model_name'] = aug_model_names.values[0]
            # Separate the first 50 rows
            first_50_rows = data.iloc[len(data) // 2:, :]
            # Identify rows where "ori_or_aug" == 'aug' and "model_name" is not NaN
            aug_model_names = first_50_rows[first_50_rows['ori_or_aug'] == 'aug']['model_name']
            # Assign model names to the first 50 rows where "ori_or_aug" == 'ori' and "model_name" is NaN
            data.loc[first_50_rows.index, 'model_name'] = aug_model_names.values[0]
            grouped = data.groupby(['diagnosis', "model_name", 'Drug', "ori_or_aug"])
            # Compute the sample count for each group in grouped
            sample_counts = data.groupby(['diagnosis', 'Drug']).size().reset_index(
                    name='Sample count')
            file_processed_data = []
            for (diagnosis, model_name, drug, ori_or_aug), group in grouped:
                filtered_group = group.dropna(subset=['ground_truth', 'prediction'])
                if len(filtered_group) < 2:
                    continue
                print(
                        f"Processing: Diagnosis={diagnosis}, Drug={drug}, Sample count={filtered_group.shape[0]}")
                reg_metrics = compute_metrics(
                        filtered_group['ground_truth'], filtered_group['prediction'])
                clf_metrics = compute_post_hoc_classification_metrics(
                        filtered_group['ground_truth'], filtered_group['prediction'])
                # Retrieve the sample count for the current diagnosis and drug
                sample_count = sample_counts[
                    (sample_counts['diagnosis'] == diagnosis) & (sample_counts['Drug'] == drug)
                    ]['Sample count'].values[0]  # Use `.values[0]` to extract the scalar value
                file_processed_data.append(
                        {
                                "Drug": drug,
                                "model_name": model_name,
                                "ori_or_aug": ori_or_aug,
                                "beta": beta,
                                "num2aug": num2aug,
                                "Drug (cancer count)": f"{drug} ({sample_count})",
                                "TCGA Classification": diagnosis,
                                "Diagnosis": diagnosis,
                                "Sample count": sample_count,
                                **clf_metrics,  # Unpack the dictionary directly
                                **reg_metrics,  # Unpack the dictionary directly
                        })
            case_processed_data_df = pd.DataFrame(file_processed_data).reset_index()
            coll_processed_data.append(case_processed_data_df)

        all_drugs_data_df = pd.concat(coll_processed_data)

        plt.figure()
        sns.barplot(all_drugs_data_df, x="Drug", y="F1-Score", hue="ori_or_aug")

        drug_cv_metric_mean_stats = compute_CV_mean_summary_statistics(
                all_drugs_data_df,
                metrics_cols=['Accuracy', 'F1-Score'],
                groupby=['Drug', 'ori_or_aug'])  ## ['Accuracy', 'Precision', 'Recall', 'F1-Score', "Spearman Correlation", "Pearson Correlation", "R2 Score"]
        """['mean_pearson_aug', 'mean_pearson_ori',..., 'Drug']"""
        drug_mean_metric_improvement = compute_improvement(
                drug_cv_metric_mean_stats,
                metrics=[ele for ele in
                         drug_cv_metric_mean_stats.columns if
                         "mean" in ele])
        # Melt the dataframe for long-form representation for plotting
        plot_df = drug_cv_metric_mean_stats.melt(
                id_vars=["Drug", "ori_or_aug"],
                value_vars=[ele for ele in drug_cv_metric_mean_stats.columns if
                            "mean" in ele],
                var_name="metric",
                value_name="score")  ## ['Drug', 'ori_or_aug', 'metric', 'score']
        #
        drug_mean_std_metrics_long = plot_df.groupby(["Drug", "ori_or_aug", "metric"]).agg(
                mean_score=("score", "mean"),
                std_score=("score", "std")
        ).reset_index()
        plot_heatmap(
                drug_mean_metric_improvement, prefix=trial_name,
                sort_col="mean_accuracy_improvement",
                save_dir=data_dir)
        plot_facetgrid(
                drug_mean_std_metrics_long,
                drug_mean_metric_improvement.sort_values(
                        "mean_accuracy_improvement", ascending=False),
                top=True, selectK=20, plot_metrics=["Accuracy", "F1-Score"], prefix=trial_name, save_dir=data_dir)
        plot_facetgrid(
                drug_mean_std_metrics_long,
                drug_mean_metric_improvement.sort_values(
                        "mean_accuracy_improvement", ascending=False),
                top=False, selectK=20, plot_metrics=["Accuracy", "F1-Score"], prefix=trial_name, save_dir=data_dir)
        drug_mean_logfc_p_df = compute_log_fold_change_and_paired_t_tests(
                drug_cv_metric_mean_stats)
        plot_slope_chart(
                drug_cv_metric_mean_stats, column="ori_or_aug", prefix=trial_name,
                value2plot="mean_accuracy",
                save_dir=data_dir)
        ## melt results_df for plotting
        # melt_results_df = all_drugs_data_df.melt(
        #         id_vars=["Drug", "ori_or_aug", "Diagnosis", "Sample count"],
        #         value_vars=["Spearman Correlation", "Pearson Correlation", "R2 Score"],
        #         var_name="metric", value_name="score")
        plot_boxstripplot_ori_and_aug_overall_drugs(
                plot_df, metric="mean_f1-score", y_col="score", save_dir=data_dir)
        compute_p_values_paired_metrics(drug_cv_metric_mean_stats)

## 2024.12.21
def load_TCGA_saved_predictons_get_vis_new():

    data_dirs = [
            r"..\results\GDSC-RES\GDSC-aug-with-TCGA\12-19T23-48-TCGA_TCGA_beta0.6_num50",
            # r"..\results\GDSC-RES\GDSC-aug-with-TCGA\12-19T23-48-TCGA_TCGA_beta0.6_num100",
            # r"..\results\GDSC-RES\GDSC-aug-with-TCGA\12-19T23-48-TCGA_TCGA_beta0.6_num200",
            # r"..\results\GDSC-RES\GDSC-aug-with-TCGA\12-19T23-48-TCGA_TCGA_beta1_num50",
            # r"..\results\GDSC-RES\GDSC-aug-with-TCGA\12-19T23-48-TCGA_TCGA_beta1_num200",
            # r"..\results\GDSC-RES\GDSC-aug-with-TCGA\12-19T23-48-TCGA_TCGA_beta2_num50",
            # r"..\results\GDSC-RES\GDSC-aug-with-TCGA\12-19T23-48-TCGA_TCGA_beta2_num200",
    ]
    all_folders_coll_processed_data = []
    raw_coll_all_drugs = []
    for data_dir in data_dirs:
        coll_processed_data = []
        csv_files = glob(path.join(data_dir, "*pred_gt_meta*csv"))
        trial_name = path.basename(data_dir).split("-")[-1]
        for file in csv_files:
            data = pd.read_csv(file)

            rename_hue = {"ori": "Original", "aug": "Augment"}
            data["ori_or_aug"] = data["ori_or_aug"].map(rename_hue)

            raw_coll_all_drugs.append(data)
            beta = path.basename(data_dir).split("_")[-2].split("beta")[-1]
            num2aug = path.basename(data_dir).split("_")[-1].split("num")[-1]

            if "model_name" in data.columns:
                grouped = data.groupby(['diagnosis', 'Drug', "ori_or_aug", "model_name"])
                sample_counts = data.groupby(
                        ['diagnosis', 'Drug', "ori_or_aug", "model_name"]).size().reset_index(
                        name='Sample count')
            else:
                grouped = data.groupby(['diagnosis', 'Drug', "ori_or_aug"])
                # Compute the sample count for each group in grouped
                sample_counts = data.groupby(
                        ['diagnosis', 'Drug', "ori_or_aug"]).size().reset_index(
                        name='Sample count')
            file_processed_data = []
            for group_keys, group in grouped:
                if "model_name" in data.columns:
                    diagnosis, drug, ori_or_aug, model_name = group_keys
                    # Retrieve the sample count for the current diagnosis and drug
                    sample_count = sample_counts[
                        (sample_counts['diagnosis'] == diagnosis) & (sample_counts['Drug'] == drug)
                        & (sample_counts['model_name'] == model_name) & (
                                sample_counts['ori_or_aug'] == ori_or_aug)
                        ]['Sample count'].values[0]  # Use `.values[0]` to extract the scalar value
                else:
                    diagnosis, drug, ori_or_aug = group_keys
                    model_name = "FNN"
                    # Retrieve the sample count for the current diagnosis and drug
                    sample_count = sample_counts[
                        (sample_counts['diagnosis'] == diagnosis) & (sample_counts['Drug'] == drug)
                        & (sample_counts['ori_or_aug'] == ori_or_aug)
                        ]['Sample count'].values[0]  # Use `.values[0]` to extract the scalar value

                print(
                        f"Processing: Diagnosis={diagnosis}, Drug={drug}, Sample count={filtered_group.shape[0]}")
                filtered_group = group.dropna(subset=['ground_truth', 'prediction'])
                if len(filtered_group) == 0:
                    continue
                reg_metrics = compute_metrics(
                        filtered_group['ground_truth'], filtered_group['prediction'])
                clf_metrics = compute_post_hoc_classification_metrics(
                        filtered_group['ground_truth'], filtered_group['prediction'])

                file_processed_data.append(
                        {
                                "Drug": drug,
                                "Diagnosis": diagnosis,
                                "model_name": model_name,
                                "ori_or_aug": ori_or_aug,
                                "beta": beta,
                                "num2aug": num2aug,
                                "Drug (cancer count)": f"{drug} ({sample_count})",
                                "TCGA Classification": diagnosis,
                                "Sample count": sample_count,
                                **clf_metrics,  # Unpack the dictionary directly
                                **reg_metrics,  # Unpack the dictionary directly
                        })
            ## get df for this drug/file
            case_processed_data_df = pd.DataFrame(file_processed_data).reset_index()
            coll_processed_data.append(case_processed_data_df)
        ## all drugs' sample-wise pred and gt and meta
        raw_coll_all_drugs_df = pd.concat(raw_coll_all_drugs, axis=0).reset_index()

        ## Plot the cancer-wise prediction vs ground truth for all drugs
        plot_cancer_grouped_pred_vs_gt_all_drugs(data_dir, raw_coll_all_drugs_df, trial_name)

        ## get all drugs/files df
        coll_processed_data_df = pd.concat(coll_processed_data, axis=0).reset_index()
        ## save the result data
        coll_processed_data_df.to_csv(path.join(data_dir, f"{trial_name}_drugs_cancer_model_summary.csv"), index=False)

        ## get all drugs/files df
        all_folders_coll_processed_data.append(coll_processed_data_df)

        uniq_models = coll_processed_data_df["model_name"].unique()
        with open(path.join(data_dir, f"{trial_name}_latex_code.txt"), "wb") as file:
            for m_j, model_name in enumerate(uniq_models):
                drug_model_results_df = coll_processed_data_df[
                    coll_processed_data_df["model_name"] == model_name]
                # Compute the mean and std for the desired metrics while keeping the "ori_or_aug" column
                desired_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                # Group by "Drug" and "ori_or_aug" and calculate mean and std
                stats = drug_model_results_df.groupby(['ori_or_aug']).agg(
                        **{f"{metric}_Mean": (metric, 'mean') for metric in desired_metrics},
                        **{f"{metric}_Std": (metric, 'std') for metric in desired_metrics}
                ).reset_index()
                # Group by Drug and "ori_or_aug" to compute statistics
                ori_stats = stats[stats['ori_or_aug'] == 'Original']
                aug_stats = stats[stats['ori_or_aug'] == 'Augment']
                # Extract metrics for LaTeX
                ori_f1 = f"${ori_stats['F1-Score_Mean'].values[0]:.2f} \\pm {ori_stats['F1-Score_Std'].values[0]:.2f}$"
                aug_f1 = f"${aug_stats['F1-Score_Mean'].values[0]:.2f} \\pm {aug_stats['F1-Score_Std'].values[0]:.2f}$"
                ori_accuracy = f"${ori_stats['Accuracy_Mean'].values[0]:.2f} \\pm {ori_stats['Accuracy_Std'].values[0]:.2f}$"
                aug_accuracy = f"${aug_stats['Accuracy_Mean'].values[0]:.2f} \\pm {aug_stats['Accuracy_Std'].values[0]:.2f}$"
                # Generate LaTeX row
                latex_row = f"{model_name}$^{{2}}$ & {ori_accuracy}  & {ori_f1}  & {aug_accuracy}  & {aug_f1} \\\n"
                print(latex_row)
                if m_j == 0:
                    header_row = f"model_name & ori_accuracy & ori_f1 &  aug_accuracy &  aug_f1 \\\n"
                    file.write(header_row)
                file.write(latex_row)

            # Define color palette for specific `value_type`
            custom_palette = {
                    "Augment": 'tab:orange',
                    "Original": 'tab:blue'
            }
            group_by_col = "Drug"
            unique_classes = drug_model_results_df["Drug"].unique()


            plt.figure(figsize=(max(len(unique_classes) * 0.45, 8), 6))
            for i, cls in enumerate(unique_classes):
                if i % 2 == 0:  # Add alternating gray background shades
                    plt.axvspan(i - 0.5, i + 0.5, color='lightgray', alpha=0.3, zorder=0)
            ax = sns.barplot(
                    data=drug_model_results_df,
                    x="Drug", y='F1-Score', hue='ori_or_aug', capsize=0.5,
                    palette=custom_palette,  ## order=unique_classes
            )
            # Customize plot
            plt.legend(loc="lower right")
            # Update the x-tick labels to include sample counts
            # Add correlation text to the top of each violin
            xtick_labels = []
            for i, group in enumerate(drug_model_results_df[group_by_col].unique()):
                xtick_labels.append(f"{group}")
            ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
            plt.legend(bbox_to_anchor=(1.02, 0.5), loc='upper left')
            # Set plot title and labels
            plt.xlabel(group_by_col)
            plt.ylabel('F1-Score')
            plt.tight_layout()
            plt.savefig(path.join(data_dir, f"0-{trial_name}-F1-Score_barplot_{model_name}.png"))

            plot_drug_cancer_pred_and_gt_heatmap(
                data_dir, model_name, raw_coll_all_drugs_df, trial_name)

            for ori_aug in ["Original", "Augment"]:
                fnn_data = drug_model_results_df[drug_model_results_df["ori_or_aug"] == ori_aug]
                # Filter data for original (ori) and augmented (aug)
                raw_gt_data = drug_model_results_df[
                    (drug_model_results_df["ori_or_aug"] == "Original") & (drug_model_results_df["output"] == "ground_truth")
                    ]
                raw_pred_data = drug_model_results_df[
                    (drug_model_results_df["ori_or_aug"] == "Original") & (drug_model_results_df["output"] == "prediction")
                    ]
                # Pivot the data to create matrices for the heatmaps
                raw_gt_heatmap_data = raw_gt_data.pivot_table(
                        index="diagnosis", columns="Drug", values="output_values", aggfunc="mean"
                )
                raw_pred_heatmap_data = raw_pred_data.pivot_table(
                        index="diagnosis", columns="Drug", values="output_values", aggfunc="mean"
                )
                raw_difference_heatmap_data = raw_pred_heatmap_data - raw_gt_heatmap_data
                # Create the heatmaps
                fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
                #
                # sns.heatmap(raw_pred_heatmap_data, ax=axes[0],
                #             cmap="coolwarm", annot=True, fmt=".2f",
                #             linewidths=0.5, cbar_kws={"label": "Difference"})
                # axes[0].set_xlabel("Drugs")
                # axes[0].set_ylabel("Diagnosis")
                # Heatmap for "ori"
                sns.heatmap(
                        raw_gt_heatmap_data,
                        ax=axes[0],
                        cmap="viridis",
                        annot=False,
                        linewidths=0.5,
                        cbar_kws={"label": "Ground Truth"},
                )
                axes[0].set_title("Ground Truth")
                axes[0].set_xlabel("Drugs")
                axes[0].set_ylabel("Diagnosis")
                # Heatmap for "aug"
                sns.heatmap(
                        raw_pred_heatmap_data,
                        ax=axes[1],
                        cmap="viridis",
                        annot=False,
                        linewidths=0.5,
                        cbar_kws={"label": "Ground Truth"},
                )
                axes[1].set_title("Prediction")
                axes[1].set_xlabel("Drugs")
                # Adjust layout
                plt.suptitle(f"Heatmaps of Ground Truth and Predictions {ori_aug}", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(path.join(data_dir, f"2-heatmap-gt-vs-pred-cancer-drugs {ori_aug}.png"))

                # Example data for a bubble chart
                # Step 1: Filter data for ground_truth
                ground_truth_data = fnn_data[fnn_data["output"] == "ground_truth"]

                # Step 2: Categorize responses based on thresholds
                # Example thresholds: Modify as needed based on your data
                def categorize_response(value):
                    if value >= 0.8:  # Complete response
                        return "Complete"
                    elif value >= 0.5:  # Partial response
                        return "Partial"
                    else:  # Non-respond
                        return "Non-respond"

                # Apply the categorization
                ground_truth_data["Response Category"] = ground_truth_data["output_values"].apply(
                        categorize_response)

                # Step 3: Aggregate data by Cancer and Drug
                response_df = ground_truth_data.groupby(
                        ["diagnosis", "Drug", "Response Category"]).size().reset_index(name="Count")

                # Step 4: Pivot to make visualization-ready (for heatmap and bar chart)
                response_pivot = response_df.pivot_table(
                        index="diagnosis", columns="Drug", values="Count", aggfunc="sum", fill_value=0
                )

                # Step 5: Normalize for response rate (optional, for certain plots)
                response_rate_df = response_df.groupby(["diagnosis", "Drug"]).apply(
                        lambda x: x.assign(ResponseRate=x["Count"] / x["Count"].sum())
                ).reset_index(drop=True)
                heatmap_response_rate = response_rate_df.pivot_table(
                        index="diagnosis", columns="Drug", values="ResponseRate", fill_value=0
                )

                response_summary_table = response_rate_df.pivot_table(
                        index=["diagnosis", "Drug"],
                        columns="Response Category",
                        values="ResponseRate",
                        aggfunc="mean",
                        fill_value=0
                ).reset_index()
                response_summary_table.rename(
                        columns={
                                "Complete": "Complete Response Rate",
                                "Partial": "Partial Response Rate",
                                "Non-respond": "Non-respond Rate"
                        },
                        inplace=True
                )
                response_summary_table = response_summary_table.sort_values(by=["diagnosis", "Drug"])

                # Prepare data for visualization
                stacked_data = response_summary_table.melt(
                        id_vars=["diagnosis", "Drug"],
                        value_vars=["Complete Response Rate", "Partial Response Rate",
                                    "Non-respond Rate"],
                        var_name="Response Category",
                        value_name="Rate"
                )

                ####OK Prepare data for heatmap
                heatmap_data = response_summary_table.pivot_table(
                        index="diagnosis",
                        columns="Drug",
                        values="Complete Response Rate"
                )

                # Plot heatmap
                plt.figure(figsize=(15, 10))
                sns.heatmap(
                        heatmap_data,
                        annot=True,
                        fmt=".2f",
                        cmap="coolwarm",
                        linewidths=0.5,
                        cbar_kws={"label": "Complete Response Rate"}
                )
                plt.title("Heatmap of Complete Response Rates")
                plt.xlabel("Drugs")
                plt.ylabel("Cancer Types")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
                # Define custom palette for response categories
                category_palette = {
                        "Complete Response Rate": "tab:green",
                        "Partial Response Rate": "tab:blue",
                        "Non-respond Rate": "tab:red"
                }
                plt.figure(figsize=(12, 8))
                for category, color in category_palette.items():
                    category_data = response_summary_table.melt(
                            id_vars=["diagnosis", "Drug"],
                            value_vars=[category],
                            var_name="Response Category",
                            value_name="Rate"
                    )
                    sns.scatterplot(
                            data=category_data,
                            x="Drug",
                            y="diagnosis",
                            size="Rate",
                            hue="Response Category",
                            palette=category_palette,
                            sizes=(1, 200),
                    )
                plt.title(f"Bubble Chart for {category}")
                plt.xlabel("Drugs")
                plt.ylabel("Cancer Types")
                plt.legend(bbox_to_anchor=[1.02, 1], loc="upper left")
                plt.tight_layout()
                plt.xticks(rotation=45, ha="right")
                plt.show()

        drug_cv_metric_mean_stats = compute_CV_mean_summary_statistics(
                drug_model_results_df,
                metrics_cols=['Accuracy', 'F1-Score'],
                ##,"Spearman Correlation", "Pearson Correlation", "R2 Score"'Precision', 'Recall',
                groupby=['Drug', 'ori_or_aug'])
        """['mean_pearson_aug', 'mean_pearson_ori',..., 'Drug']"""
        drug_mean_metric_improvement = compute_improvement(
                drug_cv_metric_mean_stats,
                metrics=[ele for ele in
                         drug_cv_metric_mean_stats.columns if
                         "mean" in ele])
        # Melt the dataframe for long-form representation for plotting
        plot_df = drug_cv_metric_mean_stats.melt(
                id_vars=["Drug", "ori_or_aug"],
                value_vars=[ele for ele in drug_cv_metric_mean_stats.columns if
                            "mean" in ele],
                var_name="metric",
                value_name="score")  ## ['Drug', 'ori_or_aug', 'metric', 'score']

        plot_heatmap(
                drug_mean_metric_improvement, prefix=trial_name+f"-{model_name}",
                sort_col="mean_accuracy_improvement",
                save_dir=data_dir)
        plot_facetgrid(
                drug_model_results_df, drug_mean_metric_improvement.sort_values(
                        "mean_accuracy_improvement", ascending=False),
                top=True, selectK=20, plot_metrics=["F1-Score"],
                prefix=trial_name + f"-{model_name}", save_dir=data_dir)
        drug_mean_logfc_p_df = compute_log_fold_change_and_paired_t_tests(
            drug_cv_metric_mean_stats)
        plot_slope_chart(
                drug_cv_metric_mean_stats, column="ori_or_aug", prefix=trial_name,
                value2plot="mean_accuracy",
                save_dir=data_dir)
        ## melt results_df for plotting
        melt_results_df = case_processed_data_df.melt(
                id_vars=["Drug", "ori_or_aug", "Diagnosis", "Sample count"],
                value_vars=["Spearman Correlation", "Pearson Correlation", "R2 Score"],
                var_name="metric", value_name="score")
        plot_boxstripplot_ori_and_aug_overall_drugs(
                plot_df, metric="mean_f1-score", y_col="score", save_dir=data_dir)
        compute_p_values_paired_metrics(drug_cv_metric_mean_stats)

    all_folders_coll_processed_data_df = pd.concat(all_folders_coll_processed_data, axis=0).reset_index()

    plt.figure()
    sns.boxplot(coll_processed_data_df, x="Drug", y="F1-Score", hue="ori_or_aug")
    plt.xticks(rotation=45, ha="right")

    plt.figure()
    sns.boxplot(coll_processed_data_df, x="Drug", y="F1-Score", hue="Diagnosis")
    plt.xticks(rotation=45, ha="right")
    plt.legend(bbox_to_anchor=[1.02, 1])
    plt.tight_layout()

    extract_label = "num2aug"

    for metrics in ["Spearman Correlation", "Pearson Correlation"]:
        for group_by_col in ["Diagnosis", "Drug"]:
            # Pivot the DataFrame so that Num2aug values appear as columns
            hyperparam_heatmap_group_by_col(
                extract_label, coll_processed_data_df, metrics=metrics,
                save_dir=data_dir, group_by=group_by_col)
        hyperparam_wise_boxplot(
            coll_processed_data_df, extract_label=extract_label,
            metrics=metrics,
            save_dir=data_dir)
        # Categorize drugs by standard deviation for subplots
        hyperparam_effect_vis_categorized_by_std(
                coll_processed_data_df, extract_label=extract_label, metrics=metrics, save_dir=data_dir)
        # Set up subplots for each category
        hyperparam_effect_vis_mean_with_fill_between(
                coll_processed_data_df, extract_label=extract_label,
                metrics=metrics, save_dir=data_dir)
        hyperparam_effect_grouped_by_max_metric_group(
                coll_processed_data_df,
                extract_label=extract_label, metrics=metrics, save_dir=data_dir)


def plot_drug_cancer_pred_and_gt_heatmap(data_dir, model_name, raw_coll_all_drugs_df, trial_name):
    melt_raw_pred_gt_df = raw_coll_all_drugs_df.melt(
            id_vars=["Drug", "ori_or_aug", "diagnosis", "model_name", "short_sample_id"],
            value_vars=["prediction", "ground_truth"],
            var_name="output", value_name="output_values")
    model_data = melt_raw_pred_gt_df[melt_raw_pred_gt_df["model_name"] == model_name]
    for ori_aug in ["Original", "Augment"]:
        fnn_data = model_data[model_data["ori_or_aug"] == ori_aug]
        # Filter data for original (ori) and augmented (aug)
        raw_gt_data = model_data[
            (model_data["ori_or_aug"] == "Original") & (
                    model_data["output"] == "ground_truth")
            ]
        raw_pred_data = model_data[
            (model_data["ori_or_aug"] == "Original") & (
                    model_data["output"] == "prediction")
            ]
        # Pivot the data to create matrices for the heatmaps
        raw_gt_heatmap_data = raw_gt_data.pivot_table(
                index="diagnosis", columns="Drug", values="output_values", aggfunc="mean"
        )
        raw_pred_heatmap_data = raw_pred_data.pivot_table(
                index="diagnosis", columns="Drug", values="output_values", aggfunc="mean"
        )
        raw_difference_heatmap_data = raw_pred_heatmap_data - raw_gt_heatmap_data
        # Create the heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

        sns.heatmap(
                raw_gt_heatmap_data,
                ax=axes[0],
                cmap="viridis",
                annot=False,
                linewidths=0.5,
                cbar_kws={"label": "Ground Truth"},
        )
        axes[0].set_title("Ground Truth")
        axes[0].set_xlabel("Drugs")
        axes[0].set_ylabel("Diagnosis")
        # Heatmap for "aug"
        sns.heatmap(
                raw_pred_heatmap_data,
                ax=axes[1],
                cmap="viridis",
                annot=False,
                linewidths=0.5,
                cbar_kws={"label": "Ground Truth"},
        )
        axes[1].set_title("Prediction")
        axes[1].set_xlabel("Drugs")
        # Adjust layout
        plt.suptitle(f"Heatmaps of Ground Truth and Predictions {ori_aug}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(path.join(data_dir, f"0-{trial_name}-Heatmap_{model_name}_{ori_aug}.png"))
        plt.show()


def plot_cancer_grouped_pred_vs_gt_all_drugs(data_dir, raw_coll_all_drugs_df, trial_name):
    # def plot_cancer_grouped_pred_vs_gt_all_drugs(data_dir, raw_coll_all_drugs_df, trial_name):
    melt_pred_gt_df = raw_coll_all_drugs_df.melt(
            id_vars=["Drug", "ori_or_aug", "diagnosis", "model_name", "short_sample_id"],
            value_vars=["prediction", "ground_truth"],
            var_name="output", value_name="output_values")
    unique_classes = melt_pred_gt_df["Drug"].unique()

    # Plotting
    fig, axes = plt.subplots(5, 4, figsize=(20, 20))  # Adjust grid size as needed
    for ax, drug in zip(axes.flatten(), unique_classes):
        drug_data = melt_pred_gt_df[
            (melt_pred_gt_df["Drug"] == drug) & (melt_pred_gt_df["model_name"] == "FNN")]
        # Compute mean ground_truth and counts for sorting and labeling
        diagnosis_stats = drug_data[drug_data["output"] == "ground_truth"].groupby(
                "diagnosis").agg(
                mean_ground_truth=("output_values", "mean"),
                count=("output_values", "size")
        ).reset_index()
        # Create a mapping for sorted diagnoses with counts
        diagnosis_stats = diagnosis_stats.sort_values(by="mean_ground_truth", ascending=True)
        diagnosis_stats["label"] = diagnosis_stats["diagnosis"] + " (" + diagnosis_stats[
            "count"].astype(str) + ")"
        # Add alternating gray background shades
        for i, label in enumerate(diagnosis_stats["label"]):
            if i % 2 == 0:  # Alternate background
                ax.axvspan(i - 0.5, i + 0.5, color='lightgray', alpha=0.3, zorder=0)
        filtered_drug_data = drug_data.dropna(subset=["output_values"])
        # Violin plot
        sns.violinplot(
                data=filtered_drug_data,
                x="diagnosis",  # Use the sorted and labeled diagnosis
                y="output_values",
                hue="output",
                ax=ax,
                palette="tab10",
                bw=0.8,  # Adjust bandwidth for smoother distribution
                inner="point",
                # cut=0  # Restrict violin tails to data range
                # order=diagnosis_stats["label"],  # Ensure sorted order
        )
        ax.set_xticklabels(diagnosis_stats["label"], ha="right", rotation=45)

        plt.legend(loc="upper left")
        ax.set_title(f"{drug}")
        ax.set_xlabel("Diagnosis")
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(path.join(data_dir, f"{trial_name}_violinplot_all_drugs.png"))
    plt.close()


def get_response_data_stats(tcga_tumor_drug_pairs):
    def generate_latex_table_from_stats(stats, caption, label, disease_col_width="4cm"):
        """
        Generate LaTeX code for a table from stats DataFrame.

        Parameters:
        - stats (pd.DataFrame): DataFrame with stats to include in the table.
        - caption (str): Caption for the table.
        - label (str): Label for referencing the table.
        - disease_col_width (str): Width for the 'uniq diseases' column in LaTeX.

        Returns:
        - str: LaTeX table code.
        """
        latex_code = r"\begin{table}[t]" + "\n"
        latex_code += r"\caption{" + caption + "}\n"
        latex_code += r"\label{" + label + "}\n"
        latex_code += r"\tabcolsep=0pt"
        latex_code += r"\begin{tabular*}{0.5\textwidth}{@{\extracolsep{\fill}}lp{" + disease_col_width + r"}ccc@{\extracolsep{\fill}}}" + "\n"
        latex_code += r"\toprule%" + "\n"
        latex_code += r"\textbf{drug name} & \textbf{uniq diseases} & \textbf{Non-response} & \textbf{Partial} & \textbf{Complete} \\\\" + "\n"

        for _, row in stats.iterrows():
            drug_name = row["drug_name"]
            uniq_diseases = row["uniq_diseases"]
            non_response = row.get("count_response_0.0", 0)
            partial = row.get("count_response_0.5", 0)
            complete = row.get("count_response_1.0", 0)
            latex_code += f"{drug_name} & {uniq_diseases} & {non_response} & {partial} & {complete} \\\\\n"

        latex_code += r"\midrule" + "\n"
        latex_code += r"\botrule" + "\n"
        latex_code += r"\end{tabular*}" + "\n"
        latex_code += r"\end{table}" + "\n"

        return latex_code

    subset = tcga_tumor_drug_pairs[tcga_tumor_drug_pairs["drug_name"].isin(list(shared_drugs))]

    # Compute the stats with unique disease codes and binary response counts
    stats = subset.groupby("drug_name").agg(
            uniq_diseases=("disease_code", lambda x: len(np.unique(x))),
            # Unique disease codes as a string
    ).reset_index()

    # Count occurrences of each binary response per drug
    binary_response_counts = subset.groupby(["drug_name", "binary_response"]).size().unstack(
        fill_value=0)

    # Rename columns for binary responses
    binary_response_counts.columns = [f"count_response_{col}" for col in
                                      binary_response_counts.columns]

    # Merge stats with binary response counts
    stats = stats.merge(binary_response_counts, on="drug_name", how="left")

    # Example usage
    caption = "Drug-wise statistics including unique diseases and binary response counts."
    label = "tab:drug_stats"

    latex_table_code = generate_latex_table_from_stats(stats, caption, label)
    # Display the LaTeX code
    print(latex_table_code)


def compare_two_folders_results():
    """
    Compare the results from two different folders and generate a summary table.
    :return:
    """
    def load_and_prepare_data(data_dirs, file_pattern, metrics_cols, groupby_cols, extract_label,
                              model_name, dir_names=["Noise", "Mix"]):
        """
        Load data from multiple directories, compute mean and statistics, and prepare for plotting.
        """
        combined_data = []
        for i, data_dir, category_prefix in zip(np.arange(len(data_dirs)), data_dirs, dir_names):
            folder_name = path.basename(data_dir)
            # Process metrics for each file in the folder
            folder_results_list = process_metrics_for_each_file_one_folder(
                    data_dir, file_pattern,
                    lambda dir: float(path.basename(dir).split('_')[-1].split("num")[-1]),
                    groupby_col=groupby_cols,
                    extract_label=extract_label,
                    if_with_clf_metrics=True,
                    return_raw_data=False
            )
            folder_results_df = pd.DataFrame(folder_results_list)
            drug_cv_metric_mean_stats = compute_CV_mean_summary_statistics(
                    folder_results_df,
                    metrics_cols=metrics_cols,
                    groupby=["Drug", "ori_or_aug", "Diagnosis", "model_name"]
            )
            # Filter for the specified model and append results
            if i > 0:
                filtered_data = drug_cv_metric_mean_stats[
                    (drug_cv_metric_mean_stats["model_name"] == model_name) & (
                            drug_cv_metric_mean_stats["ori_or_aug"].str.contains("aug"))]
            else:
                filtered_data = drug_cv_metric_mean_stats[
                    drug_cv_metric_mean_stats["model_name"] == model_name]
            # Add a folder-specific label to distinguish conditions
            filtered_data["ori_or_aug"] = filtered_data["ori_or_aug"].apply(
                    lambda
                        x: f"{x}{category_prefix}" if x == "aug" else "ori" if i == 0 else f"{x}{category_prefix}"
            )
            combined_data.append(filtered_data)
        # Combine data from all folders
        return pd.concat(combined_data, axis=0)

    def plot_facetgrid_with_three_categories(data, metrics, top, prefix, selectK, save_dir):
        """
        Plot the metrics comparison for the top or bottom improved drugs with three categories.
        """
        # Sort drugs by improvement and select top or bottom K
        drugs = data["Drug"].unique()[:selectK] if top else data["Drug"].unique()[-selectK:]
        subset = data[data["Drug"].isin(drugs)]
        subset = subset[subset["metric"].isin(metrics)]
        # Define plot title
        title = f"Top {selectK} improved drugs" if top else f"Bottom {selectK} worsened drugs"
        # Create FacetGrid with the additional hue categories
        g = sns.catplot(
                data=subset,
                x="Drug", y="mean_score", hue="ori_or_aug", col="metric",
                kind="bar", ci="sd", col_wrap=1, height=4, aspect=1.5,
                capsize=0.25,
                sharex=False, sharey=False
        )
        # Rotate x-tick labels for better readability
        for ax in g.axes.flat:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.grid(alpha=0.3)
        # Add legend and customize layout
        g.add_legend(title="Condition")
        g.set_titles(title + " {col_name}")
        plt.tight_layout()
        # Save the plot
        plt.savefig(path.join(save_dir, f"facetgrid_{prefix}_{title}.png"))
        plt.show()

    # Example Usage
    data_dirs = [
            r"..\results\GDSC-RES\GDSC-aug-with-TCGA\12-27T10-38-GDSC+TCGA_TCGA_noise0.10_num400",
            r"..\results\GDSC-RES\GDSC-aug-with-TCGA\12-22T00-27-GDSC+TCGA_TCGA_beta1_num400",
    ]
    metrics_cols = ['Spearman Correlation', 'Pearson Correlation', 'Accuracy', 'F1-Score']
    groupby_cols = ["Drug", "diagnosis", "model_name"]
    model_name = "FNN"
    file_pattern = "*with_pred_gt_me*csv"
    # Load and prepare data
    combined_data = load_and_prepare_data(
            data_dirs, file_pattern, metrics_cols, groupby_cols, "Num2aug", model_name,
            dir_names=["Noise", "Mix"])
    # Melt the combined data for long-form representation for plotting
    plot_df = combined_data.melt(
            id_vars=["Drug", "ori_or_aug", "Diagnosis", "model_name"],
            value_vars=["mean_pearson", "mean_spearman", "mean_accuracy", "mean_f1-score"],
            var_name="metric", value_name="score"
    )
    drug_mean_std_metrics_long = plot_df.groupby(
            ["Drug", "ori_or_aug", "Diagnosis", "model_name", "metric"]
    ).agg(
            mean_score=("score", "mean"),
            std_score=("score", "std")
    ).reset_index()
    # Plot FacetGrid with three categories
    plot_facetgrid_with_three_categories(
            drug_mean_std_metrics_long, metrics=["mean_spearman"],
            top=True, prefix="comparison", selectK=20, save_dir=data_dirs[-1]
    )

    # Group by Drug, Diagnosis, and ori_or_aug, then calculate the mean for each metric
    mean_data = combined_data.groupby(['Drug', 'ori_or_aug'], as_index=False)[
        ['mean_spearman', 'mean_f1-score']
    ].mean()

    # Separate data into original and augmented cases
    original_data_mean = mean_data[mean_data['ori_or_aug'] == 'original']
    aug_data_mean = mean_data[mean_data['ori_or_aug'] != 'original']

    # Recalculate the number of drugs with metrics over 0.6
    spearman_over_06_mean = mean_data[mean_data['mean_spearman'] > 0.4].shape[0]
    f1_over_06_mean = mean_data[mean_data['mean_f1-score'] > 0.5].shape[0]

    # Recalculate improvement counts
    improved_spearman_mean = 0
    improved_f1_score_mean = 0

    for drug in mean_data['Drug'].unique():
        ori_spearman = original_data_mean[original_data_mean['Drug'] == drug][
            'mean_spearman'].mean()
        ori_f1_score = original_data_mean[original_data_mean['Drug'] == drug][
            'mean_f1-score'].mean()

        aug_spearman = aug_data_mean[aug_data_mean['Drug'] == drug]['mean_spearman'].mean()
        aug_f1_score = aug_data_mean[aug_data_mean['Drug'] == drug]['mean_f1-score'].mean()

        if aug_spearman > ori_spearman:
            improved_spearman_mean += 1
        if aug_f1_score > ori_f1_score:
            improved_f1_score_mean += 1

    # Updated descriptive statistics
    stats_mean = mean_data.groupby('ori_or_aug')[['mean_spearman', 'mean_f1-score']].describe()
    stats_mean.to_csv(
        path.join(
            data_dir,
            f"stats_mean_0.05vs0.1-aug200_over0.5f1{f1_over_06_mean}_over0.4Sp{spearman_over_06_mean}.csv"))


if __name__ == '__main__':

    plot_mode = "load_analyze_plot_metrics_of_one_folder"
    if plot_mode == "load_saved_file_for_performance_volcano_drugs":
        load_csv_and_compute_and_plot_volcano_drug_level()
    elif plot_mode == "load_saved_file_for_performance_volcano_group":
        load_csv_and_compute_and_plot_volcano_group_level(
            data_dir=r"..\results\GDSC-RES\models\leave-cell-line-out\12-13T02-01_beta5.0_6clsZ",
            prefix="GDSC-RES"
        )
    elif plot_mode == "plot_overall_pred_vs_gt_scatter_with_contour":
        data_dir = r"..\results\GDSC-RES\models\leave-cell-line-out\12-13T02-01_beta5.0_6clsZ"
        plot_overall_pred_vs_gt_scatter_with_contour(data_dir)
    elif plot_mode == "visualize_metrics_across_folders":
        visualize_metrics_across_folders()
    elif plot_mode == "load_analyze_plot_metrics_of_one_folder":
        load_analyze_plot_metrics_of_one_folder()
    elif plot_mode == "compare_two_folders_results":
        compare_two_folders_results()


