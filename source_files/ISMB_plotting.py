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
from sklearn.metrics import mean_squared_error, r2_score
from plotting_utils import generate_dark_colors
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix




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
        for case, case_data in zip(["ori", "aug"], [ori_data, aug_data]):
            if case == "aug":
                prefix = "DA with " + beta_str
            else:
                prefix = "No DA"

            pearson_corr, _ = pearsonr(case_data["Ground Truth"], case_data["Predictions"])
            sp_corr, p_value = spearmanr(case_data["Ground Truth"], case_data["Predictions"])
            r2 = r2_score(case_data["Ground Truth"], case_data["Predictions"])
            corr_str = f"Pearson r: {pearson_corr:.2f}\nSpearman r: {sp_corr:.2f}\nR2:{r2:.2f}"

            # Calculate point density
            xy = np.vstack([case_data["Ground Truth"], case_data["Predictions"]])
            density_dict[case] = gaussian_kde(xy)(xy)
            # Create scatter plot with density-based coloring
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(
                    case_data["Ground Truth"],
                    case_data["Predictions"],
                    c=density_dict[case],
                    cmap='viridis',
                    s=20,
                    edgecolor='gray'
            )
            plt.text(
                    0.03, 0.95, corr_str,
                    transform=plt.gca().transAxes, fontsize=15, verticalalignment='top',
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
            )
            # Add diagonal line for perfect predictions
            min_val = min(case_data["Ground Truth"].min(), case_data["Predictions"].min())
            max_val = max(case_data["Ground Truth"].max(), case_data["Predictions"].max())
            plt.plot(
                    [min_val, max_val], [min_val, max_val], color='k', linestyle='--',
                    label='Perfect Prediction')
            plt.colorbar(scatter, label='Density')
            plt.title(f"Prediction vs Ground Truth ({prefix})")
            plt.xlabel("Ground Truth")
            plt.ylabel("Predictions")
            plt.grid(alpha=0.3)
            plt.savefig(path.join(data_dir, f"0-{prefix}-all-{case}_scatter_with_density.png"))
            plt.close()


def load_saved_pred_gt_files(data_dir, file_patten="No*.csv"):
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
    spearman_corr, _ = spearmanr(ground_truth, predictions)
    pearson_corr, _ = pearsonr(ground_truth, predictions)
    r2 = r2_score(ground_truth, predictions)
    rmse = mean_squared_error(ground_truth, predictions, squared=False)
    return {
            "Spearman Correlation": spearman_corr,
            "Pearson Correlation": pearson_corr,
            "R2 Score": r2,
            "RMSE": rmse
    }


#
# # Function to calculate performance metrics
# def compute_metrics(ground_truth, predictions):
#     spearman_corr, _ = spearmanr(ground_truth, predictions)
#     pearson_corr, _ = pearsonr(ground_truth, predictions)
#     r2 = r2_score(ground_truth, predictions)
#     rmse = mean_squared_error(ground_truth, predictions, squared=False)
#     return pd.DataFrame({
#             "Spearman Correlation": spearman_corr,
#             "Pearson Correlation": pearson_corr,
#             "R2 Score": r2,
#             "RMSE": rmse
#     })


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
#     }


def visualize_metrics_across_folders():
    def load_saved_pred_gt_files(data_dir, file_patten="No*.csv"):
        files = glob(path.join(data_dir, file_patten))
        # Rewriting the process to calculate p-values and generate a volcano plot correctly
        # Load all ori and aug files
        ori_files = [file for file in files if 'ori' in file]
        aug_files = [file for file in files if 'aug' in file]
        ori_data = []
        aug_data = []
        # Process ori files
        if len(ori_files) > 0:
            for file in ori_files:
                data = pd.read_csv(file)
                data['ori_or_aug'] = 'ori'  # Assign 'ori' for original files
                ori_data.append(data)
            ori_data_df = pd.concat(ori_data)
        else:
            ori_data_df = None
        # Process aug files
        if len(aug_files) > 0:
            for file in aug_files:
                data = pd.read_csv(file)
                data['ori_or_aug'] = 'aug'  # Assign 'aug' for augmented files
                aug_data.append(data)
            aug_data_df = pd.concat(aug_data)
        else:
            aug_data_df = None

        return ori_data_df, aug_data_df

    def compute_metrics(ground_truth, predictions):
        """
        Compute performance metrics between ground truth and predictions.
        Parameters:
            ground_truth (pd.Series): Ground truth values.
            predictions (pd.Series): Predicted values.
        Returns:
            dict: Dictionary containing the computed metrics.
        """
        spearman_corr, _ = spearmanr(ground_truth, predictions)
        pearson_corr, _ = pearsonr(ground_truth, predictions)
        r2 = r2_score(ground_truth, predictions)
        rmse = mean_squared_error(ground_truth, predictions, squared=False)
        return {
                "Spearman Correlation": spearman_corr,
                "Pearson Correlation": pearson_corr,
                "R2 Score": r2,
                "RMSE": rmse
        }

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
                                    save_dir=None, group_by="Cancer Type"):
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
    plt.title(f"Heatmap of {metrics} Grouped by {group_by}")
    plt.xlabel(f"Parameter {extract_label}")
    plt.ylabel(group_by.capitalize())
    plt.tight_layout()
    plt.savefig(path.join(save_dir, f"0-heatmap_{metrics}_grouped_by_{group_by}.png"))
    plt.close()


def hyperparam_wise_boxplot(plot_data, extract_label="Num2aug",
                            metrics="Spearman Correlation",
                            save_dir="./"):
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
    plt.title(f"{metrics} Distribution Across {extract_label}")
    plt.xlabel(f"Parameter {extract_label}")
    plt.ylabel(metrics)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path.join(save_dir, f"0-params-{extract_label}-boxplot.png"))
    plt.close()


def hyperparam_effect_grouped_by_max_metric_group(no_da_data, extract_label="Num2aug",
                                                  metrics="Spearman Correlation", save_dir=None):
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

        ax.set_title(f"Drugs with Max Performance at {extract_label}={hyperparam}")
        ax.set_xlabel(f"Parameter {extract_label}")
        ax.set_ylabel(metrics)
        ax.legend(title="Drug", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path.join(save_dir, f"0-{extract_label}-drugs_max_sep.png"))
    plt.close()


def hyperparam_effect_vis_mean_with_fill_between(no_da_data, extract_label="Num2aug",
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
            bbox_to_anchor=(0.0, 0),
            loc="upper center",
            ncol=10,
            frameon=False
    )
    plt.xlabel(extract_label)
    plt.ylabel(metrics)
    plt.tight_layout()
    plt.savefig(path.join(save_dir, f"0-{extract_label}-drugs_fill_between-{metrics}.png"))
    plt.close()


def hyperparam_effect_vis_categorized_by_std(no_da_data, extract_label="Num2aug",
                                             metrics="Spearmanorrelation", save_dir=None):
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

        ax.set_title(f"Spearman Correlation - {category}")
        ax.set_xlabel(extract_label)
        ax.set_ylabel(metrics)
        ax.legend(title="Drugs", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(path.join(save_dir, f"0-{extract_label}-drugs_sep_std-{metrics}.png"))
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
        agg_dict[f'mean_{metric_name}'] = (metric, 'mean')
        agg_dict[f'std_{metric_name}'] = (metric, 'std')

    # return coll_metric_ori_w_aug.groupby(groupby).agg(
    #         mean_spearman=('Spearman Correlation', 'mean'),
    #         std_spearman=('Spearman Correlation', 'std'),
    #         mean_pearson=('Pearson Correlation', 'mean'),
    #         std_pearson=('Pearson Correlation', 'std'),
    #         mean_r2=('R2 Score', 'mean'),
    #         std_r2=('R2 Score', 'std')
    # ).reset_index()
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


def plot_facetgrid(grouped, sort_improvement_df, prefix="", top=True, selectK=5, save_dir="./"):
    # Select top or bottom improved drugs
    drugs = sort_improvement_df["Drug"].unique()[:selectK] if top else sort_improvement_df[
                                                                           "Drug"].unique()[
                                                                       -selectK:]
    subset = grouped[grouped["Drug"].isin(drugs)]

    title = f"Top {selectK} improved drugs" if top else f"Bottom {selectK} worsened drugs"

    # Create the catplot
    g = sns.catplot(
            data=subset,
            x="Drug", y="mean_score", hue="ori_or_aug", col="metric",
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


def compute_log_fold_change_and_paired_t_tests(summary_stats):
    from scipy.stats import ttest_rel
    pivoted = summary_stats.pivot_table(
            index='Drug', columns='ori_or_aug',
            values=['mean_spearman', 'mean_pearson', 'mean_r2']
    )
    pivoted.columns = ['_'.join(col).strip() for col in pivoted.columns.values]
    pivoted = pivoted.reset_index()
    for metric in ['mean_spearman', 'mean_pearson', 'mean_r2']:
        pivoted[f'logfc_{metric}'] = np.log2(
                pivoted[f'{metric}_aug'] / pivoted[f'{metric}_ori']
        )
    p_values = {}
    for metric in ['mean_spearman', 'mean_pearson', 'mean_r2']:
        ori = pivoted[f'{metric}_ori']
        aug = pivoted[f'{metric}_aug']
        t_stat, p_val = ttest_rel(ori, aug)
        p_values[f'p_value_{metric}'] = -np.log10(p_val)
    for metric in ['mean_spearman', 'mean_pearson', 'mean_r2']:
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
def load_saved_csvs_get_stats_of_one_folder(ori_data_dir, aug_data_dir):
    # ori_file_dir = (r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active projects\GexMix-ISMB\results\GDSC-RES\models\leave-cell-line-out\12-16T23-38_fnn_beta0.5_6zsco_aug100")
    ori_files = glob(path.join(ori_data_dir, "No*ori*csv"))

    # aug_file_dir = r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active projects\GexMix-ISMB\results\GDSC-RES\models\leave-cell-line-out\12-16T23-38_fnn_beta1.0_6zsco_aug100"
    aug_files = glob(path.join(aug_data_dir, "No*aug*csv"))

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
                    if len(group) < 2:
                        continue
                    print(
                            f"Processing: Diagnosis={diagnosis}, Drug={drug}, Sample count={group.shape[0]}")
                    metrics = compute_metrics(group['Ground Truth'], group['Predictions'])
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


def load_analyze_plot_metrics_of_one_folder(file_fomat="pickle"):
    """
    This is overarching function
    for getting the metrics of all drugs in one folder from either the
    saved pickle files (load_saved_pickle_get_stats_of_one_folder)or the saved csv files
    load_saved_csv_get_stats_of_one_folder.
    :return:
    """
    ori_data_dir = r"\results\GDSC-RES\models\leave-cell-line-out\12-16T23-38_fnn_beta0.5_6zsco_aug100"
    aug_data_dir = r"\results\GDSC-RES\models\leave-cell-line-out\12-16T23-38_fnn_beta1.0_6zsco_aug100"

    # Load the saved pickle files
    if file_fomat == "pickle":
        results_df = load_saved_pickle_get_stats_of_one_folder(ori_data_dir, aug_data_dir)
    elif file_fomat == "csv":
        results_df = load_saved_csvs_get_stats_of_one_folder(ori_data_dir, aug_data_dir)

    """Get mean over all samples of one drug
    ['Drug', 'ori_or_aug', 'mean_spearman', 'std_spearman',  'mean_pearson',
       'std_pearson', 'mean_r2', 'std_r2']"""
    drug_cv_metric_mean_stats = compute_CV_mean_summary_statistics(
        results_df,
        groupby=['Drug', 'ori_or_aug'])

    """['mean_pearson_aug', 'mean_pearson_ori', 'mean_r2_aug', 'mean_r2_ori',
             'mean_spearman_aug', 'mean_spearman_ori', 'spearman_improvement',
             'pearson_improvement', 'r2_improvement', 'Drug']"""
    drug_mean_metric_improvement = compute_improvement(drug_cv_metric_mean_stats)

    # Melt the dataframe for long-form representation for plotting
    plot_df = drug_cv_metric_mean_stats.melt(
            id_vars=["Drug",  "ori_or_aug"],
            value_vars=["mean_pearson", "mean_spearman"],
            var_name="metric", value_name="score")  ## ['Drug', 'ori_or_aug', 'metric', 'score']

    #
    drug_mean_std_metrics_long = plot_df.groupby(["Drug", "ori_or_aug", "metric"]).agg(
            mean_score=("score", "mean"),
            std_score=("score", "std")
    ).reset_index()
    plot_heatmap(drug_mean_metric_improvement, sort_col="spearman_improvement", prefix="", save_dir=aug_data_dir)
    plot_facetgrid(
            drug_mean_std_metrics_long, drug_mean_metric_improvement.sort_values("spearman_improvement", ascending=False),
            top=True, prefix="", save_dir=aug_data_dir)
    plot_facetgrid(
            drug_mean_std_metrics_long, drug_mean_metric_improvement.sort_values("spearman_improvement", ascending=False),
            top=False, prefix="", save_dir=aug_data_dir)
    drug_mean_logfc_p_df = compute_log_fold_change_and_paired_t_tests(drug_cv_metric_mean_stats)
    plot_slope_chart(drug_cv_metric_mean_stats, column="ori_or_aug", prefix="", value2plot="mean_spearman", save_dir=aug_data_dir)

    ## melt results_df for plotting
    melt_results_df = results_df.melt(
        id_vars=["Drug", "ori_or_aug", "Diagnosis", "Sample count"],
        value_vars=["Spearman Correlation", "Pearson Correlation", "R2 Score"],
        var_name="metric", value_name="score")
    plot_boxstripplot_ori_and_aug_overall_drugs(melt_results_df, metric="mean_pearson", prefix="", y_col="score", save_dir=aug_data_dir)

    compute_p_values_paired_metrics(drug_cv_metric_mean_stats)


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

    # Annotate the p-value
    x1, x2 = 0, 1  # Position of 'ori' and 'aug' on the x-axis
    y_max = plot_df[y_col].max() + 0.1  # Height for the annotation
    plt.plot([x1, x2], [y_max, y_max], color='black', linewidth=1.5)  # Line connecting groups

    if p_value <= 0.05:
        if p_value < 0.001:
            sig_str = "***"
        elif p_value < 0.01:
            sig_str = "**"
        else:
            sig_str = "*"
        plt.text(
                (x1 + x2) / 2, y_max + 0.5, f'{sig_str}', ha='center', va='bottom', fontsize=12)

    # Customize the plot
    plt.title(f"Comparison of Ori and Aug Drugs\n{prefix}", fontsize=14)
    plt.ylabel("Metric Mean", fontsize=12)
    plt.xlabel("Condition", fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(path.join(save_dir, f"0-{prefix}-{metric}_ori_vs_aug_stripboxplot.png"))
    plt.close()



def compute_p_values_paired_metrics(drug_cv_metric_mean_stats):
    pivot_paired_metrics_df = drug_cv_metric_mean_stats.pivot_table(
            index='Drug', columns='ori_or_aug', values=['mean_spearman', 'mean_pearson', 'mean_r2'])
    # Perform paired t-tests
    t_stat_spearman, p_val_spearman = ttest_rel(
            pivot_paired_metrics_df['mean_spearman']['aug'],
            pivot_paired_metrics_df['mean_spearman']['ori'])
    t_stat_pearson, p_val_pearson = ttest_rel(
            pivot_paired_metrics_df['mean_pearson']['aug'],
            pivot_paired_metrics_df['mean_pearson']['ori'])
    t_stat_r2, p_val_r2 = ttest_rel(
            pivot_paired_metrics_df['mean_r2']['aug'], pivot_paired_metrics_df['mean_r2']['ori'])
    print(f"Spearman Score: t-stat={t_stat_spearman:.3f}, p-value={p_val_spearman:.3e}")
    print(f"Pearson Score: t-stat={t_stat_pearson:.3f}, p-value={p_val_pearson:.3e}")
    print(f"R2 Score: t-stat={t_stat_r2:.3f}, p-value={p_val_r2:.3e}")
    wilcoxon_spearman = wilcoxon(
            pivot_paired_metrics_df['mean_spearman']['aug'],
            pivot_paired_metrics_df['mean_spearman']['ori'])
    wilcoxon_pearson = wilcoxon(
            pivot_paired_metrics_df['mean_pearson']['aug'],
            pivot_paired_metrics_df['mean_pearson']['ori'])
    wilcoxon_r2 = wilcoxon(
            pivot_paired_metrics_df['mean_r2']['aug'], pivot_paired_metrics_df['mean_r2']['ori'])
    print(f"Spearman Score Wilcoxon: p-value={wilcoxon_spearman.pvalue:.3e}")
    print(f"Pearson Score Wilcoxon: p-value={wilcoxon_pearson.pvalue:.3e}")
    print(f"R2 Score Wilcoxon: p-value={wilcoxon_r2.pvalue:.3e}")


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


def load_saved_TCGA_model_summary_get_visualization():
    """
    Load saved drug response data from GDSC to TCGA and visualize
    :return:
    """
    data_dir = r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active projects\GexMix-ISMB\results\GDSC-RES\GDSC-aug-with-TCGA\12-19T10-49-GDSC+TCGA_TCGA_beta0.6_num50"
    files = glob(path.join(data_dir, "*summary.csv"))
    # Rewriting the process to calculate p-values and generate a volcano plot correctly

    aug_data = []
    for file in files:
        data = pd.read_csv(file)
        aug_data.append(data)
    aug_data_df = pd.concat(aug_data)

    # save_dir2 = path.dirname(filename)
    # loaded_all_drug_data = pd.read_csv(filename)
    for model_name in aug_data_df["model_name"].unique():
        fnn_df = aug_data_df[aug_data_df["model_name"] == model_name]
        # Group by drug and original/augment to calculate mean accuracy
        grouped_data = fnn_df.groupby(['drug', 'ori_or_aug'])['accuracy'].agg(
                ['mean', 'std']).reset_index()
        # Add improvement (augment - original)
        original_acc = grouped_data[grouped_data['ori_or_aug'] == 'original']
        augment_acc = grouped_data[grouped_data['ori_or_aug'] == 'augment']
        improvement = augment_acc.set_index('drug')['mean'] - original_acc.set_index('drug')['mean']
        grouped_data['improvement'] = grouped_data['drug'].map(improvement)
        # Create a barplot with error bars
        plt.figure(figsize=(14, 8))
        sns.barplot(
                data=grouped_data,
                x='drug',
                y='mean',
                hue='ori_or_aug',
                palette='pastel',
                ci="sd",
        )
        # Add error bars
        for index, row in grouped_data.iterrows():
            plt.errorbar(
                    x=index // 2 + (0.2 if row['ori_or_aug'] == 'ori' else -0.2),
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
        plt.savefig(fr"{data_dir}\{model_name}mean_accuracy_by_drug_and_data_type.png")


def post_hoc_classification_metrics(ground_truth, predictions, thresholds=[0.25, 0.75]):
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
    # ground_truth = filtered_group['ground_truth']
    # predictions = filtered_group['prediction']
    # thresholds = [0.25, 0.75]
    # Map predictions to discrete classes using thresholds
    discrete_predictions = np.digitize(predictions, bins=thresholds)
    class_mapping = {0: 0, 1: 0.5, 2: 1}  # Map indices to discrete classess
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
    #### Compute classification metrics Compute regression-like metrics (MSE, MAE)
    mse = np.mean((predictions - ground_truth) ** 2)
    mae = np.mean(np.abs(predictions - ground_truth))
    # Combine all metrics
    metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Confusion Matrix": confusion.tolist(),
            "MSE": mse,
            "MAE": mae
    }

    return metrics

def load_TCGA_saved_predictons_get_vis():
    data_dir = r"C:\Users\DiyuanLu\Documents\1-Helmholtz-projects\0-active projects\GexMix-ISMB\results\GDSC-RES\GDSC-aug-with-TCGA\12-19T10-49-GDSC+TCGA_TCGA_beta0.6_num50"
    aug_files = glob(path.join(data_dir, "*pred_gt_meta*csv"))
    coll_processed_data = []
    # for case_data in (aug_files):
    #     case_processed_data = []
    #     if len(case_data) > 0:
    for file in aug_files:
        data = pd.read_csv(file)
        case_name = path.basename(data_dir).split("-")[-1]
        grouped = data.groupby(['diagnosis', 'Drug', "ori_or_aug"])
        # Compute the sample count for each group in grouped
        sample_counts = data.groupby(['diagnosis', 'Drug']).size().reset_index(
                name='Sample count')
        file_processed_data = []
        for (diagnosis, drug, ori_or_aug), group in grouped:
            filtered_group = group.dropna(subset=['ground_truth', 'prediction'])
            if len(filtered_group) < 2:
                continue

            print(
                    f"Processing: Diagnosis={diagnosis}, Drug={drug}, Sample count={filtered_group.shape[0]}")
            reg_metrics = compute_metrics(
                    filtered_group['ground_truth'], filtered_group['prediction'])
            clf_metrics = post_hoc_classification_metrics(
                    filtered_group['ground_truth'], filtered_group['prediction'])
            # Retrieve the sample count for the current diagnosis and drug
            sample_count = sample_counts[
                (sample_counts['diagnosis'] == diagnosis) & (sample_counts['Drug'] == drug)
                ]['Sample count'].values[0]  # Use `.values[0]` to extract the scalar value
            file_processed_data.append(
                    {
                            "Drug": drug,
                            "ori_or_aug": ori_or_aug,
                            "Drug (cancer count)": f"{drug} ({sample_count})",
                            "TCGA Classification": diagnosis,
                            "Diagnosis": diagnosis,
                            "Sample count": sample_count,
                            **clf_metrics,  # Unpack the dictionary directly
                            **reg_metrics,  # Unpack the dictionary directly
                    })
        case_processed_data_df = pd.DataFrame(file_processed_data).reset_index()
        coll_processed_data.append(case_processed_data_df)
    coll_processed_data_df = pd.concat(coll_processed_data, axis=0).reset_index()

    plt.figure()
    sns.boxplot(coll_processed_data_df, x="Drug", y="F1-Score", hue="ori_or_aug")
    plt.xticks(rotation=45, ha="right")

    plt.figure()
    sns.boxplot(coll_processed_data_df, x="Drug", y="F1-Score", hue="Diagnosis")
    plt.xticks(rotation=45, ha="right")
    plt.legend(bbox_to_anchor=[1.02, 1])
    plt.tight_layout()


    drug_cv_metric_mean_stats = compute_CV_mean_summary_statistics(
            coll_processed_data_df, metrics_cols=['Accuracy', 'Precision', 'Recall', 'F1-Score',
                                                  "Spearman Correlation", "Pearson Correlation", "R2 Score"],
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
            var_name="metric", value_name="score")  ## ['Drug', 'ori_or_aug', 'metric', 'score']
    #
    drug_mean_std_metrics_long = plot_df.groupby(["Drug", "ori_or_aug", "metric"]).agg(
            mean_score=("score", "mean"),
            std_score=("score", "std")
    ).reset_index()
    plot_heatmap(
            drug_mean_metric_improvement, prefix=case_name, sort_col="mean_accuracy_improvement",
            save_dir=data_dir)
    plot_facetgrid(
            drug_mean_std_metrics_long,
            drug_mean_metric_improvement.sort_values("mean_accuracy_improvement", ascending=False),
            top=True, prefix=case_name, save_dir=data_dir)
    plot_facetgrid(
            drug_mean_std_metrics_long,
            drug_mean_metric_improvement.sort_values("mean_accuracy_improvement", ascending=False),
            top=False, prefix=case_name, save_dir=data_dir)
    drug_mean_logfc_p_df = compute_log_fold_change_and_paired_t_tests(drug_cv_metric_mean_stats)
    plot_slope_chart(
            drug_cv_metric_mean_stats, column="ori_or_aug", prefix=case_name, value2plot="mean_accuracy",
            save_dir=data_dir)
    ## melt results_df for plotting
    melt_results_df = coll_processed_data_df.melt(
            id_vars=["Drug", "ori_or_aug", "Diagnosis", "Sample count"],
            value_vars=["Spearman Correlation", "Pearson Correlation", "R2 Score"],
            var_name="metric", value_name="score")
    plot_boxstripplot_ori_and_aug_overall_drugs(
            plot_df, metric="mean_f1-score", y_col="score", save_dir=data_dir)
    compute_p_values_paired_metrics(drug_cv_metric_mean_stats)

if __name__ == '__main__':

    plot_mode = "load_saved_TCGA_drug_response_get_visualization"
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
    elif plot_mode == "load_saved_TCGA_drug_response_get_visualization":
        load_saved_TCGA_model_summary_get_visualization()


