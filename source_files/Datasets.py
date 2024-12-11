import random
import pickle
import joblib  # for
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
from textwrap import wrap
from os import path, makedirs, listdir
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from glob import glob

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.model_selection import train_test_split

from plotting_utils import (getmeta2trackdict, generate_pastel_colors,
                            interactive_bokeh_all_meta_in1_subplots,
                            interactive_bokeh_with_select_test_for_DGEA,
                            interactive_bokeh_multi_subplots_unify_colorcoding,
                            visualize_data_with_meta,
                            plot_individual_cancer_project)  # to keep the preprocessed meta data consistent with the meta we need in
# the dataset


def get_filename_dict():
    filename_dict = {
            "TCGA": {
                    # "filename":
                    #          '../data/1_filter_CCLE_with_TCGA/filtered_TCGA_with_shared_gene_between_CCLE_10374_17713.csv',
                    #      "meta_filename":
                    #      '../data/1_filter_CCLE_with_TCGA/final-resorted-samples-based-HiSeqV2
                    #      -new.csv'},
                    "filename": '../data/tcga_10374_5000_std.pickle',
                    "meta_filename": '../data/tcga_10374_5000_std.pickle',
                    "drug_response":
                        "../data/GDSC_drug_response/all_clin_XML_curated_annotation_matched_v1_alex.csv"
            },
            "CCLE": {
                    # "filename":
                    # '../data/1_filter_CCLE_with_TCGA/filtered_CCLE_with_shared_gene_between_TCGA_1248_17713_renameCol.csv',
                    # "meta_filename":
                    # '../data/1_filter_CCLE_with_TCGA/CCLE_sample_info_with_imputed_TCGA_labels_and_unified_site_names_step3_1248.csv'},

                    "filename": '../data/ccle_1248_5000_std.pickle',
                    "meta_filename": '../data/ccle_1248_5000_std.pickle'
            },
            "GDSC": {
                    "filename": r"..\data\GDSC_drug_response\GDSC_gex_w_filtered_meta_(959, 19434)_processed9.pickle",
                    "meta_filename": r"..\data\GDSC_drug_response\GDSC_gex_w_filtered_meta_(959, 19434)_processed9.pickle"},
            "combat_3":{
                    # "filename": r"../data/GDSC_drug_response/combat_site+diag_TCGA+GDSC+CCLE_3.pickle",
                    # "meta_filename": r"../data/GDSC_drug_response/combat_site+diag_TCGA+GDSC+CCLE_3.pickle",../results/GDSC-RES\combat_site+diag_3sets_dataset+site_3.pickle
                    "filename": r"..\data\GDSC_drug_response\combat_site+diag_3sets_dataset+site_3.pickle",
                    "meta_filename": r"..\data\GDSC_drug_response\combat_site+diag_3sets_dataset+site_3.pickle",
            }

    }
    return filename_dict


class BasicGExDataset:

    def __init__(self, data_filename, meta_filename, args, verbose: bool):
        self.data_filename = data_filename
        self.meta_filename = meta_filename
        self.args = args
        self.raw_data = None
        self.meta_data = None
        self.metric2track_dict = getmeta2trackdict()

        if verbose:
            self.num2load = 20
        else:
            self.num2load = None

        self.load_data_and_info_given_filenames()

    def load_data_and_info_given_filenames(self):
        """
        This function need to deal with multple dataset meta-data columns and get the
        corresponding columns for each
        given unified feature columns
        # TCGA: [10374, Name+17713],
        # CCLE [1248, Name+17713], originally 1249, discarded one
        :param data_filename:
        :param meta_filename:
        :param if_to_cluster:
        :param data_root:
        :return:
        """
        _, file_extension = path.splitext(self.data_filename)
        file_extension = file_extension.lower()

        if file_extension == "csv" and self.data_filename != self.meta_filename:
            # load original data where raw data and meta data are separate
            self.raw_data = pd.read_csv(self.data_filename, nrows=self.num2load)
            self.meta_data = pd.read_csv(self.meta_filename, nrows=self.num2load)

        elif file_extension == "pickle" and self.data_filename == self.meta_filename:
            with open(self.data_filename, "rb") as file:
                loaded_pickle_data = pickle.load(file)
                self.raw_data = loaded_pickle_data["rnaseq"]
                self.meta_data = loaded_pickle_data["meta"]

        assert len(self.raw_data) == len(
                self.meta_data), "Numbers of samples don't match! between data and metadata"

    def check_stats(self):
        """Check statistics of the loaded data."""
        if self.raw_data is None or self.meta_data is None:
            print("Data has not been loaded yet.")
            return

        print("Statistics of the loaded data:")
        print("Raw data shape:", self.raw_data.shape)
        print("Metadata:", self.meta_data)
        for key in self.metric2track_dict.keys():
            print(f"{key}: [{np.unique(self.meta_data[key])}]")


class GExDataset():

    def __init__(self, data_filename, meta_filename, name: str, verbose=True, save_dir="./"):
        """

        :param data_filename:
        :param meta_filename:
        :param args:
        :param feature_selection_basis_fn: a pre-computed file for the corresponding dataset
        """
        self.data_filename = data_filename
        self.meta_filename = meta_filename
        # self.num_classes = args.num_classes
        # self.feature_selection_basis_fn = feature_selection_basis_fn
        # self.args = args
        self.gex_data = None
        self.meta_data = None
        self.dataset_name = name
        self.save_dir = save_dir

        if verbose:
            self.nrows2load = 50
        else:
            self.nrows2load = None  # all rows
        # need to edit given need
        self.meta2track_dict = getmeta2trackdict()

    def load_gex_data(self):
        """
        :param mode: ["data", "meta", "all"]
        # TCGA: [10374, Name+17713],
        # CCLE [1248, Name+17713], originally 1249, discarded one
        :return:
        """
        if self.gex_data is None:
            _, file_extension = path.splitext(self.data_filename)
            if file_extension == ".csv":
                self.gex_data = pd.read_csv(
                    self.data_filename,
                    nrows=self.nrows2load)  # [10374, Name+17713]
            elif file_extension == ".pickle" and self.data_filename == self.meta_filename:  #
                # already presaved all together in pickle
                with open(self.data_filename, "rb") as f:
                    pickle_data = pickle.load(f)
                    self.gex_data = pickle_data["rnaseq"]
                    self.meta_data = pickle_data["meta"]
                    self.meta_data["diagnosis"] = np.array(self.meta_data["diagnosis"]).astype(str)

                    if "dataset_name" not in self.meta_data.columns:
                        self.meta_data["dataset_name"] = [self.dataset_name] * len(self.meta_data)

                    if self.dataset_name != "TCGA":
                        self.meta_data["cell_line_name"] = self.meta_data["stripped_cell_line_name"]

        if self.meta_data is None:
            print("Metadata is not loaded yet!")
        else:
            assert len(self.gex_data) == len(self.meta_data), "Numbers of TCGA sample don't match!"
            print("Gex and Meta data are all loaded!")

        # df_group_normalized, df_ignored, meta_data_filtered, meta_data_ignored =
        # self.grouped_zscore_normalization(
        #     self.gex_data, self.meta_data, label_column='diagnosis', labels_to_ignore=[])

    def load_meta_data(self):
        if self.meta_data is None:
            _, file_extension = path.splitext(self.data_filename)
            if file_extension == ".csv":
                self.meta_data = pd.read_csv(self.meta_filename, nrows=self.nrows2load)
            elif file_extension == ".pickle" and self.data_filename == self.meta_filename:  #
                # already presaved all together in pickle
                with open(self.data_filename, "rb") as f:
                    pickle_data = pickle.load(f)
                    self.gex_data = pickle_data["rnaseq"]
                    self.meta_data = pickle_data["meta"]
                    self.meta_data["diagnosis"] = np.array(self.meta_data["diagnosis"]).astype(str)

                    if "dataset_name" not in self.meta_data.columns:
                        self.meta_data["dataset_name"] = [self.dataset_name] * len(self.meta_data)
                    if self.dataset_name != "TCGA":
                        self.meta_data["cell_line_name"] = self.meta_data["stripped_cell_line_name"]

        if self.gex_data is None:
            print("Gex data is not loaded yet!")
        else:
            assert len(self.gex_data) == len(
                    self.meta_data), "Numbers of TCGA sample don't match!"
        print(f"{self.dataset_name} is ok")

        self.label_dict = {}
        for key in self.meta2track_dict.keys():
            self.label_dict[key] = self.meta_data[key].values

        self.check_stats_of_dataset()

    def check_stats_of_dataset(self):
        self.unique_cancer = np.unique(
            self.label_dict["diagnosis"].astype(str))  # there are 0 or 0.0
        self.cancer_classes = len(self.unique_cancer)
        # Count the frequency of each unique value in the 'Category' column
        value_counts = self.meta_data['diagnosis'].value_counts()

        # Plot the bar plot
        plt.figure(figsize=[15, 6])
        plt.bar(value_counts.index, value_counts.values)

        # Annotate counts on top of the bars
        for i, count in enumerate(value_counts.values):
            plt.text(i, count + 0.1, str(count), ha='center')

        plt.xlabel(f'Cancer types TCGA')
        plt.ylabel('Frequency')
        plt.title(f'Bar Plot of {self.dataset_name} Category Frequencies')
        plt.xticks(rotation=90)
        plt.savefig(
            path.join(
                self.save_dir, f"original_data_stats_{len(self.label_dict['diagnosis'])}.png"))
        plt.close()

    def load_data_with_meta_data(self):
        self.load_gex_data()
        self.load_meta_data()

        assert len(self.gex_data) == len(
                self.meta_data), "Numbers of TCGA sample don't match!"

    def print_columns_with_few_unique_values(self, meta_data, cutoff=10):
        for column in meta_data.columns:
            unique_values = meta_data[column].value_counts()
            # unique_values = meta_data[column].unique()
            if len(unique_values) < cutoff and len(unique_values) > 1:
                print(f"{unique_values}")
                print(f"---------------------------")

    # high level data set
    def load_data_into_dataset(self, args):
        """
        Load from the mat file
        tag sampleID
        create tf datasets
        :param args:
        :return:
        """

        if args.num_genes == 1024:
            with open(
                    "../data/1_filter_CCLE_with_TCGA/full_data_labels_gene_11622_1024_zscore0"
                    ".pickle",
                    "rb") as file:
                related_labels_dict = pickle.load(file)
                normalized_data = related_labels_dict["rnaseq"]
                del related_labels_dict["rnaseq"]
                args.num_genes = normalized_data.shape[1]
                args.data_shape = [args.num_genes, 1, 1]

        else:
            self.related_labels_dict, self.normalized_data = (
                    self.load_filtered_TCGA_and_CCLE_dataset(
                args))


    def harmonize_meta_data_for_datasets(self, loaded_data, data_sample_info):
        """
        This function SHOULD deal with multiple datasets meta data handling
        :param data_sample_info:
        :param dataset_name:
        :param loaded_data:
        :return:# need to edit given need
        self.meta2track_dict = {"source_labels": {"dtype": np.int32, "values": []},
                                "site_labels": {"dtype": np.float32, "values": []},
                                "diagnosis_labels_b4_impute": {"dtype": str, "values": []},
                                "diagnosis": {"dtype": str, "values": []},
                                "tumor_percent": {"dtype": np.float32, "values": []},
                                "normal_percent": {"dtype": np.float32, "values": []},
                                "sample_id": {"dtype": np.float32, "values": []},
                                "tumor_stage": {"dtype": str, "values": []},
                                }
        """
        # TCGA and CCLE shared columns
        site_col_name = \
        [ele for ele in ["gdc_cases.project.primary_site", "added_transformed_lineage"] if ele in
         data_sample_info.columns][0]
        disesae_col_name = \
        [ele for ele in ["gdc_cases.project.project_id", "added_tcga_labels_b4_impute"] if ele in
         data_sample_info.columns][0]  # 'TCGA-ESCA
        disesae_labels_imputed_col_name = \
        [ele for ele in ["diagnosis", "added_tcga_labels"] if ele in
         data_sample_info.columns][0]  # ESCA
        tumor_stage = \
        [ele for ele in ["gdc_cases.diagnoses.tumor_stage", "additional_info"] if ele in
         data_sample_info.columns][0]  # ESCA
        # [print(col, np.unique(data_sample_info[col], return_counts=True)) for col in
        # data_sample_info.columns]
        if self.dataset_name == "ccle":  # unify the column name of genes between T
            loaded_data.columns = [ele.split(" (")[0] for ele in loaded_data.columns]
            data_sample_info.insert(
                    data_sample_info.shape[1], "tumor_percent",
                    100 * np.ones(len(data_sample_info)))
            data_sample_info.insert(
                    data_sample_info.shape[1], "normal_percent", np.zeros(len(data_sample_info)))
            # add the following two cols to match the format of TCGA dataset
            tumor_percent = "tumor_percent"
            normal_percent = "normal_percent"
            sample_id = "DepMap_ID"

            # correct one misclassification
            mis_clf_ind = np.where(data_sample_info["stripped_cell_line_name"] == "HCC1588")[0]
            data_sample_info.at[mis_clf_ind[0], "added_tcga_labels_b4_impute"] = "NSCLC"
            data_sample_info.at[mis_clf_ind[0], "added_tcga_labels"] = "NSCLC"
        if self.dataset_name == "tcga":
            tumor_percent = "cgc_slide_percent_tumor_nuclei"
            normal_percent = "cgc_slide_percent_normal_cells"
            sample_id = "sample_id"
        new_column_names = {
                site_col_name: 'primary_site',
                disesae_col_name: 'diagnosis_b4_impute',
                disesae_labels_imputed_col_name: "diagnosis",
                tumor_percent: 'tumor_percent',  # newly added for downstream task
                normal_percent: 'normal_percent',
                tumor_stage: "tumor_stage",
                sample_id: 'sample_id',
        }
        data_sample_info.rename(columns=new_column_names, inplace=True)
        data_sample_info['diagnosis_b4_impute'] = [ele.split("-")[-1] for ele in
                                                   data_sample_info['diagnosis_b4_impute']]
        if "Unnamed: 0" in loaded_data.columns:
            loaded_data.drop(["Unnamed: 0"], axis=1, inplace=True)
        if "Unnamed: 0" in data_sample_info.columns:
            data_sample_info.drop(["Unnamed: 0"], axis=1, inplace=True)

        return data_sample_info


class TCGAClinicalDataProcessor:
    def __init__(self, dataset, tcga_clinical_filename, save_dir="./"):
        """
        load TCGA clinical response data, then TCGA gene expression data, and process it
        :param dataset:
        :param tcga_clinical_filename:
        """
        self.dataset = dataset
        self.tcga_clinical_filename = tcga_clinical_filename
        self.save_dir = save_dir
        self.meta_all = self.dataset.meta_data.copy()
        self.gex_all = self.dataset.gex_data.copy()
        self.num_genes = self.gex_all.shape[1]

    def get_matched_gex_meta_and_response(self):
        # load TCGA clinical data
        tcga_drug_response_df = pd.read_csv(self.tcga_clinical_filename)
        tcga_drug_response_df.insert(
            0, "short_sample_id", tcga_drug_response_df["bcr_patient_barcode"])

        self.gex_all.insert(0, "sample_id", self.meta_all["sample_id"])
        self.gex_all.insert(0, "short_sample_id", self.meta_all["short_sample_id"])
        self.gex_all.insert(0, "diagnosis", self.meta_all["diagnosis"])
        if "cell_line_name" not in self.gex_all.columns:
            self.gex_all.insert(2, "cell_line_name", self.meta_all["stripped_cell_line_name"])
        self.meta_all["cell_line_name"] = self.meta_all["stripped_cell_line_name"]

        self.meta_clinical_from_tcga = self.meta_all[
            self.meta_all["short_sample_id"].isin(tcga_drug_response_df["short_sample_id"])
        ]
        self.matched_response_with_tcga = tcga_drug_response_df[
            tcga_drug_response_df["short_sample_id"].isin(self.meta_clinical_from_tcga["short_sample_id"])
        ]
        self.gex_clinical_from_tcga = self.gex_all.loc[self.meta_clinical_from_tcga.index]

        self.matched_response_with_tcga = self._harmonize_drug_names_get_stats()  ## embedded method

        return self.gex_clinical_from_tcga, self.meta_clinical_from_tcga, self.matched_response_with_tcga

    def _harmonize_drug_names_get_stats(self):
        self.matched_response_with_tcga['drug_name'] = self.matched_response_with_tcga['drug_name'].str.lower()
        # Fix the specific values using .loc
        self.matched_response_with_tcga.loc[
            self.matched_response_with_tcga['drug_name'] == "cpt 11", 'drug_name'] = "cpt11"

        need_correct_drug_names = {
                "5-fluorouracil": ["5 fluorouracil", "5-flourouracil", "5fu", "5 fu",
                                   "5 fluorouracil+leucovorin",
                                   "5 fluorouracilum", "5 fluorouracilum+leucovorin",
                                   "5- fu", "5-flourouraci", "5-fluorourac",
                                   "5-fluorouracil + leucovorin",
                                   "5-fluorouracilum", "flourouracil", "fluorouracil",
                                   "5-fluorouracil", "5-flurouracil", "5-fu", "5-fu + leulov",
                                   "5-fu+ etoposidium",
                                   "5fluorouracil+leucovorin"],
                "letrozole": ["letrozol", "letrozole", "letrozole (femara)", "letrozolum"],
                "leuprolide": ["leuprolide acetate", "leuprorelin", "leuprolide"],
                "tamoxifen": ["tamoxifen", "tamoxifen (novadex)", "tamoxifen citrate",
                              "tamoxiphen+anastrazolum",
                              "tamoxiphene", "tamoxiphene+anastrozolum"],
                "temozolomide": ["temodar", "temozolomide", "temozolamide", "themozolomide"],
                "capecitabine": ["xeloda", "capecitabine"],
                "irinotecan": ["irinotecan", "cpt-11"],
                "folfiri": ["folfiri", "folfiri/avastin"],
        }

        for unified_name in need_correct_drug_names.keys():
            for alias in need_correct_drug_names[unified_name]:
                self.matched_response_with_tcga.loc[
                    self.matched_response_with_tcga['drug_name'] == alias, 'drug_name'] = unified_name

    def get_unique_drug_counts(self):
        # Step 1: Get counts of each drug
        self.drug_counts = self.matched_response_with_tcga['drug_name'].value_counts().reset_index()
        # Rename the columns to 'drug' and 'counts'
        self.drug_counts.columns = ['drug_name', 'total_count']
        return self.drug_counts

    def get_unique_response_counts(self):
        # Step 2: Merge with the binary_response data to get counts per response category
        # First, get a DataFrame grouped by drug_name and binary_response
        self.response_counts = self.matched_response_with_tcga.groupby(
                ['drug_name', 'binary_response']).size().unstack(
                fill_value=0).reset_index()
        # Rename columns as required
        self.response_counts.columns = ['drug_name'] + [f"{resp}_count" for resp in
                                                   self.response_counts.columns[1:]]
        # Step 3: Merge the total counts with the specific response counts
        drug_stats = pd.merge(self.drug_counts, self.response_counts, on='drug_name', how='left')
        # drug_stats.to_csv(path.join(save_dir, f"TCGA_drug_stats.csv"))
        return drug_stats

    def get_shared_drugs(self, TCGA_drug_counts, gdsc1_dataset):
        shared_drugs = set(
            TCGA_drug_counts[TCGA_drug_counts['total_count'] > 20]['drug_name']
        ).intersection(set(gdsc1_dataset.gdsc_df['Drug Name'].apply(lambda x: x.lower())))
        return shared_drugs



class GExMix:

    def __init__(self, dataset=None, gex_data=None, meta_data=None, label_dict=None, cancer_classes=None,
                 if_include_original=True, num2aug=200, num2mix=2,
                 beta_param=5, if_use_remix=True, if_stratify_mix=True):
        """
                Initialize the GExMix class with flexibility to take either a GExDataset object
                or direct inputs for gene expression data and metadata.

                Args:
                    dataset (GExDataset, optional): Dataset object containing gex_data, meta_data, etc.
                    gex_data (np.ndarray, optional): Gene expression data.
                    meta_data (pd.DataFrame, optional): Metadata associated with the gene expression data.
                    label_dict (dict, optional): Dictionary mapping labels for the data.
                    cancer_classes (int, optional): Number of cancer classes.
                    if_include_original (bool): Whether to include the original data in augmentation.
                    num2aug (int): Number of augmented samples to generate.
                    num2mix (int): Number of samples to mix for augmentation.
                    beta_param (float): Beta, larger->more uniform
                    if_use_remix (bool): Whether to use remix augmentation.
                    if_stratify_mix (bool): Whether to stratify mixing by class.
                """
        # Handle inputs flexibly
        if dataset is not None:
            self.data = dataset.gex_data
            self.meta_data = dataset.meta_data
            self.label_dict = dataset.label_dict
            self.cancer_classes = dataset.cancer_classes
        elif gex_data is not None and meta_data is not None:
            self.data = gex_data
            self.meta_data = meta_data
            self.label_dict = label_dict if label_dict is not None else {}
            self.cancer_classes = cancer_classes if cancer_classes is not None else 0
        else:
            raise ValueError("You must provide either a GExDataset or both gex_data and meta_data.")

        self.data = dataset.gex_data
        self.meta_data = dataset.meta_data
        self.label_dict = dataset.label_dict
        self.if_include_original = if_include_original
        self.cancer_classes = dataset.cancer_classes
        self.num2aug = num2aug
        self.num2mix = num2mix
        self.if_use_remix = if_use_remix
        self.if_stratify_mix = if_stratify_mix
        self.beta_param = beta_param

        self.get_int_label_name_dicts(key="diagnosis")

        # Turn the int-labels into one-hot encoding
        self.label_dict["tcga_hot_labels"] = np.eye(self.cancer_classes)[
            self.label_dict["tcga_int_labels"]]
        self.uniq_classes, self.uniq_counts = np.unique(
                np.argmax(self.label_dict["tcga_hot_labels"], axis=1),
                return_counts=True)

        # get the original data there first
        self.overall_labels_dict = {key: [] for key in self.label_dict.keys()}
        self.overall_mixed_data = []
        if self.if_include_original:
            self.overall_mixed_data.append(self.data)
            for key in self.label_dict.keys():  # first record the original
                self.overall_labels_dict[key].append(self.label_dict[key])

    def reencode_cancer_types(self, meta_data_df, cancer_type_column='cancer_type'):
        """
        Function to extract cancer types from metadata and reencode them into categories.

        Parameters:
        meta_data_df (pd.DataFrame): DataFrame containing the metadata
        cancer_type_column (str): The column name that contains the cancer types

        Returns:
        pd.DataFrame: DataFrame with an additional column for encoded cancer types
        dict: Mapping of original cancer types to their encoded values
        """
        from sklearn.preprocessing import LabelEncoder
        if cancer_type_column not in meta_data_df.columns:
            raise ValueError(f"Column '{cancer_type_column}' not found in the DataFrame")

        # Extract cancer types
        cancer_types = meta_data_df[cancer_type_column]

        # Initialize LabelEncoder
        label_encoder = LabelEncoder()

        # Fit and transform the cancer types into categorical values
        encoded_cancer_types = label_encoder.fit_transform(cancer_types)

        # Add the encoded values as a new column in the DataFrame
        meta_data_df.insert(1, 'tcga_int_labels', encoded_cancer_types)

        # Create a mapping of original cancer types to their encoded values
        mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

        return meta_data_df, mapping

    def get_int_label_name_dicts(self, key="diagnosis"):
        self.uniq_diagnosis = np.unique(self.label_dict[key].astype(str))
        self.label_dict[key] = self.label_dict[key].astype(str)
        self.diagnosis2int = {}
        self.int2diagnosis = {}
        for jj, diag in enumerate(self.uniq_diagnosis):
            self.diagnosis2int[diag] = jj
            self.int2diagnosis[jj] = diag
        self.label_dict["tcga_int_labels"] = np.array(
                [self.diagnosis2int[ele] for ele in list(self.label_dict[key])])

    def check_data_type(self, data):
        if isinstance(data, pd.Series):
            if pd.api.types.is_numeric_dtype(data):
                return "numerical"
            elif pd.api.types.is_string_dtype(data):
                return "string"
            else:
                return "other"
        elif isinstance(data, np.ndarray):
            if np.issubdtype(data.dtype, np.number):
                return "numerical"
            elif np.issubdtype(data.dtype, np.str_) or np.issubdtype(
                    data.dtype, np.object_):
                return "string"
            else:
                return "other"
        else:
            raise TypeError("Input should be a pandas Series or a numpy ndarray")

    def initialize_aug_label_dict(self):
        mixed_labels_dict = {}
        self.array_keys = []
        self.string_keys = []
        for key, value in self.label_dict.items():
            if self.check_data_type(value) == "numerical":
                self.array_keys.append(key)
                if len(self.label_dict[key].shape) > 1:
                    mixed_labels_dict[key] = np.zeros(
                            (self.num2aug, self.label_dict[
                                key].shape[1]))
                elif len(self.label_dict[key].shape) == 1:
                    mixed_labels_dict[key] = np.zeros((self.num2aug))
            elif self.check_data_type(value) == "string":
                self.string_keys.append(key)
                mixed_labels_dict[key] = []
        return mixed_labels_dict

    def initialize_aug_label_dict_from_random(self, metadf, num2aug=200, keys4mix=["diagnosis"]):
        mixed_labels_dict = {}
        self.array_keys = []
        self.string_keys = []
        for key in keys4mix:
            if self.check_data_type(metadf[key]) == "numerical":
                self.array_keys.append(key)
                mixed_labels_dict[key] = np.zeros((num2aug))
            elif self.check_data_type(metadf[key]) == "string":
                self.string_keys.append(key)
                mixed_labels_dict[key] = []
            else:
                self.string_keys.append(key)
                mixed_labels_dict[key] = []
        return mixed_labels_dict

    def get_single_class_original(self, target_name: str):
        target_inds = np.where(self.label_dict["diagnosis"] == target_name)[0]
        single_class_pure_feature = self.data.values[target_inds]
        single_class_label_dict = {}
        for key in self.label_dict.keys():
            single_class_label_dict[key] = self.label_dict[key][target_inds]

        return single_class_pure_feature, single_class_label_dict, target_inds

    def augment_single_class(self, target_name: str, target_features=None, target_label_dict=None):
        """
        given a target class, to use either a specific class or random classes to augment it.
        :param pure_feature:
        :param lables2aug_dict: dict of values need to be augmented, labels should all be one-hot
        format for mixing
        :param target_label: int
        :param num2aug: int
        :param num2mix: int
        :return:
        """

        # Create a new dictionary dictB and copy values from dictA
        mixed_labels_dict = self.initialize_aug_label_dict()

        if target_features is None and target_label_dict is None:
            target_features, target_label_dict, target_inds = self.get_single_class_original(
                target_name)
            full_feature = self.data.values
        else:
            target_features = target_features
            target_label_dict = target_label_dict
            full_feature = target_features

        mix_ids_pairs_base = np.random.choice(
            np.arange(target_features.shape[0]),
            self.num2aug * self.num2mix,
            replace=True).reshape(-1, self.num2mix)

        # Generate the mixing pair indices and corresponding weights
        mix_ids_pairs, probability_vectors = self.generate_mix_inds_with_weights(
            mix_ids_pairs_base,
            target_label_dict, target_name, alpha=self.beta_param)

        # mixing samples with generated weights
        mixed_features, mixed_labels_dict = self.update_mixed_features_and_label_dict(
            mix_ids_pairs,
            mixed_labels_dict, probability_vectors, full_feature,
            target_label_dict, keys4mix=["tcga_hot_labels", "tumor_percent", "normal_percent"])

        return mixed_features, mixed_labels_dict

    def plot_save_mixing_probability_stats(self, probability_vector, prefix="", save_dir="./"):
        plt.figure(figsize=(10, 6))
        plt.hist(probability_vector, bins=30, edgecolor='black')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.title(f'Bar Plot of Mixing Probability Distribution (beta {self.beta_param})')
        plt.savefig(
            path.join(
                save_dir, f"{prefix}_mixing_probability_stats_{len(self.label_dict['diagnosis'])}.png"))
        plt.close()

    def augment_random(self, aug_by_column: str, target_features=None,
                       target_label_dict=None, if_stratify=False, num2aug=200, random_state=99,
                       keys4mix=["tcga_hot_labels", "tumor_percent", "normal_percent"],
                       if_include_original=True, save_dir="./"):
        """
        given a target class, to use either a specific class or random classes to augment it.
        :param target_features: data array of pure gene expression, no meta
        :param lables2aug_dict: dict of values need to be augmented, labels should all be one-hot
        format for mixing
        :param target_label: int
        :param num2aug: int
        :param num2mix: int
        :return:
        """
        full_feature = target_features
        try:
            full_feature = target_features.reset_index(drop=True).values
            target_label_dict = target_label_dict.reset_index(drop=True)
        except:
            print("cant reset_index")

        mix_ids_pairs = self.generate_balance_indices_for_mixing(
                pd.DataFrame(target_label_dict), aug_by_column, num2aug=num2aug)
        # mix_ids_pairs = self.generate_random_indices_for_mixing(
        #         pd.DataFrame(target_label_dict), aug_by_column, num2aug=num2aug)

        # Generate the mixing pair indices and corresponding weights
        probability_vectors = self.augment_generate_mixing_weights(
                alpha=self.beta_param, num2aug=len(mix_ids_pairs))

        self.plot_save_mixing_probability_stats(probability_vectors[:, 0],
                                                prefix=f"{self.num2aug}-beta{self.beta_param}", save_dir=save_dir)

        # Remove rows with NaN values
        valid_probability_vectors = probability_vectors[~np.isnan(probability_vectors).any(axis=1)]
        valid_mix_pairs = mix_ids_pairs[~np.isnan(probability_vectors).any(axis=1)]

        assert len(valid_probability_vectors) == len(
                valid_mix_pairs), "number of mix_id_pair not matching prob vector"
        # Create a new dictionary dictB and copy values from dictA
        mixed_labels_dict = self.initialize_aug_label_dict_from_random(
                target_label_dict, num2aug=valid_probability_vectors.shape[0],
                keys4mix=keys4mix)

        # mixing samples with generated weights
        mixed_features, mixed_labels_dict = self.update_mixed_features_and_label_dict(
                valid_mix_pairs,
                mixed_labels_dict, valid_probability_vectors, full_feature,
                target_label_dict, keys4mix=keys4mix)

        if if_include_original:
            mixed_features = np.concatenate((full_feature, mixed_features), axis=0)
            for key in mixed_labels_dict.keys():
                mixed_labels_dict[key] = np.concatenate((target_label_dict[key],
                                                         mixed_labels_dict[key]))

        return mixed_features, mixed_labels_dict

    def generate_balance_indices_for_mixing(self, meta_df, label_column, num2aug=200):
        """
        generate same number of new samples for each class. the mixing samples are drawn form a
        balanced pool
        :param meta_df:
        :param label_column:
        :param num2aug:
        :return:
        """

        grouped = meta_df.groupby(label_column)

        uniq_diseases = meta_df[label_column].value_counts()
        # Calculate the number of samples needed per group
        num_samples_per_group = num2aug

        # Initialize an empty list to collect sampled indices for the first column
        first_col_indices = []
        second_col_indices = []
        second_col_pool_indices = []
        # Sample from each group for the first column
        for name, group in grouped:
            first_col_indices.append(
                    np.random.choice(group.index, num_samples_per_group, replace=True)
            )
            for name, group in grouped:
                second_col_pool_indices.append(
                        np.random.choice(
                            group.index,
                            np.int32(num_samples_per_group / len(uniq_diseases)),  # evenly get samples from each group
                            replace=True)
                )
            second_col_indices.append(
                    np.random.choice(
                        np.concatenate(second_col_pool_indices), num_samples_per_group,
                        replace=True)
            )

        # Flatten the list of sampled indices for the first column
        first_col_indices = np.concatenate(first_col_indices)
        second_col_indices = np.concatenate(second_col_indices)

        # Combine the first and second column indices
        mix_ids_pairs_base = np.column_stack((first_col_indices, second_col_indices))

        return mix_ids_pairs_base

    def generate_random_indices_for_mixing(self, meta_df, label_column, num2aug=200):
        """
        generate same number of new samples for each class. the mixing samples are drawn form a
        balanced pool
        :param meta_df:
        :param label_column:
        :param num2aug:
        :return:
        """
        first_col_indices = np.random.choice(meta_df.index, num2aug, replace=True)

        second_col_indices = np.random.choice(meta_df.index, num2aug, replace=True)

        # Combine the first and second column indices
        mix_ids_pairs_base = np.column_stack((first_col_indices, second_col_indices))

        return mix_ids_pairs_base

    def augment_generate_mixing_weights(self, alpha=5, num2aug=200):
        """
        given target_label to generate samples. take mix_ids_pairs and replace the first column
        with the target_label
        sample inds with highest mixing weight
        This can be done with multiple datasets
        :param lables2aug_dict: labels should all be one-hot encoding
        :param mix_ids_pairs:
        :param num2aug:
        :param num2mix:
        :param target_label:
        :return:
        """
        # Generate row-wise probability vectors using Dirichlet distribution
        probability_vectors = np.random.dirichlet(np.ones(self.num2mix) * alpha, size=num2aug)

        # Sort each row by putting max values at the first index
        probability_vectors = probability_vectors[
            np.arange(probability_vectors.shape[0])[:, np.newaxis], np.argsort(
                    -probability_vectors, axis=1)]

        # Normalize the probabilities to ensure they sum up to 1
        probability_vectors /= probability_vectors.sum(axis=1, keepdims=True)

        return probability_vectors

    def generate_mix_inds_with_weights(self, mix_ids_pairs, label_dict, target_name: str,
                                       alpha=0.63):
        """
        given target_label to generate samples. take mix_ids_pairs and replace the first column
        with the target_label
        sample inds with highest mixing weight
        This can be done with multiple datasets
        :param lables2aug_dict: labels should all be one-hot encoding
        :param mix_ids_pairs:
        :param num2aug:
        :param num2mix:
        :param target_label:
        :return:
        """
        # Generate row-wise probability vectors using Dirichlet distribution
        probability_vectors = np.random.dirichlet(np.ones(self.num2mix) * alpha, size=self.num2aug)
        updated_mix_ids_pairs = mix_ids_pairs

        if target_name is None:  # target label is not specified, then no need to normalize
            probability_vectors = probability_vectors
            updated_mix_ids_pairs = mix_ids_pairs
        else:  # when specify a base sample with given target_label
            target_class_inds = \
                np.where(label_dict["diagnosis"] == target_name)[0]

            updated_mix_ids_pairs[:, 0] = np.random.choice(target_class_inds, self.num2aug)

            # Sort each row by putting max values at the first index
            probability_vectors = probability_vectors[
                np.arange(probability_vectors.shape[0])[:, np.newaxis], np.argsort(
                        -probability_vectors, axis=1)]
            # 1st col with the highest weights
            # probability_vectors[:, 0] += probability_vectors[:, -1]  # make the difference of
            # the base weight more obvious
            # Normalize the probabilities to ensure they sum up to 1
            probability_vectors /= probability_vectors.sum(axis=1, keepdims=True)

        return updated_mix_ids_pairs, probability_vectors

    def update_mix_inds_with_minority_in2sources(self, mix_ids_pairs,
                                                 target_class_inds, num_sources=2):
        """
        get the inds in the mixing configuration considering:
        1. keep the target_class labels
        2. oversample the minority source of the target class
        :param args:
        :param lables2aug_dict:
        :param mix_ids_pairs:
        :param num2aug:
        :param num2mix:
        :param target_class_inds: inds of the target class, over all sources
        :return:
        """
        # initialize target indices placeholder
        target_inds_ph = np.empty((mix_ids_pairs.shape[0], num_sources))
        # get the index of the target sample as the backbone for mixing, # replace the first
        # sample inds with the samples
        # from the target class
        target_inds_ph[:, 0] = np.random.choice(target_class_inds, self.num2aug)
        # get inds of minority source of the target class
        uniq_source, source_count = np.unique(
                np.argmax(
                        self.label_dict["source_labels"],
                        axis=1)[target_class_inds],
                return_counts=True)

        if len(uniq_source) == num_sources:  # all sources with the target label are present
            minority_source = uniq_source[
                np.argsort(source_count)[0]]  # get the index of the minority source
            minority_inds = target_class_inds[
                np.where(
                    self.label_dict["source_labels"][target_class_inds][:,
                    minority_source] == 1)[0]]
            random_minority_inds = np.random.choice(minority_inds, self.num2aug)
            # second col is the oversampled minority source of the target label
            target_inds_ph[:, 1] = random_minority_inds
        else:  # some sources are missing, then mix some other samples from the missing source.
            print(
                    f"{self.label_dict['diagnosis'][target_class_inds][0]} only has source from "
                    f"{uniq_source} ")
            random_minority_inds = np.random.choice(
                    np.arange(len(self.label_dict["source_labels"])), self.num2aug)
            # second col is the oversampled minority source of the target labels
            target_inds_ph[:, 1] = random_minority_inds

        # only consider generall all indices from target-label, and the minority source of the
        # target-label,
        # that's why the magic number np.eye(2)
        random_inds = np.eye(num_sources)[np.random.choice(num_sources, self.num2aug)].astype(
                np.int32)
        # Use advanced indexing to shuffle columns within each row
        shuffled_target_inds_ph = target_inds_ph[
            np.arange(len(target_inds_ph))[:, np.newaxis], random_inds]
        # the first col is with the biggest mixing weight
        if self.num2mix == 2:
            # only use the first column from the shuffled_target_inds_ph
            mix_ids_pairs[:, 0] = shuffled_target_inds_ph[:, 0]
        elif self.num2mix > 2:
            mix_ids_pairs[:, 0] = shuffled_target_inds_ph[:, 0]
            rand_i = np.random.choice(np.arange(1, self.num2mix), 1)
            mix_ids_pairs[:, rand_i[0]] = shuffled_target_inds_ph[:, 1]
        return mix_ids_pairs

    # TODO: perform DA per cancer type, per data source
    def update_mixed_features_and_label_dict(self, mix_ids_pairs, mixed_labels_dict,
                                             probability_vectors, all_feature, label_dict,
                                             keys4mix=["tcga_hot_labels", "source_labels"]):
        mixed_features = np.zeros((probability_vectors.shape[0], all_feature.shape[1]))

        for idx in range(self.num2mix):
            mixed_features += np.multiply(
                    all_feature[mix_ids_pairs[:, idx]],
                    probability_vectors[:, idx][:, np.newaxis])
            for key in keys4mix:
                if key in self.array_keys:
                    # Create mixed labels using the same mix factors
                    if len(label_dict[key].shape) == 2:
                        mixed_labels_dict[key] += np.multiply(
                                label_dict[key][mix_ids_pairs[:, idx]].values,
                                probability_vectors[:, idx][:, np.newaxis])
                    elif len(label_dict[key].shape) == 1:
                        print(f"{idx}, {key}")
                        mixed_labels_dict[key] += np.multiply(
                                label_dict[key][mix_ids_pairs[:, idx]].values,
                                probability_vectors[:, idx])

        # For string labels, simply take the majority label
        for str_key in self.string_keys:
            mixed_labels_dict[str_key] += list(label_dict[str_key][mix_ids_pairs[:, 0]])
            mixed_labels_dict[str_key] = np.array(mixed_labels_dict[str_key])


        return mixed_features, mixed_labels_dict


class DatasetVisualizer:
    """
    visualize a merged dataframe with the last num_genes columns are gene expression data, and the first part are meta data
    """

    def __init__(self, merged_df, num_genes, interested_meta_cols, save_dir, if_plot_eda=False):
        self.merged_df = merged_df
        self.num_genes = num_genes
        self.interested_meta_cols = interested_meta_cols
        self.save_dir = save_dir
        self.projection = None
        self.features_df = merged_df.iloc[:, -self.num_genes:]
        self.only_meta_df = merged_df.iloc[:, 0:self.num_genes]
        self.if_plot_eda = if_plot_eda

    def visualize_original_data(self, drug, drug_count, vis_mode, timestamp, cmap="jet",
                                figsize=[8, 6]):
        """Visualize the original data."""
        self.projection = visualize_data_with_meta(
                self.features_df.values,
                self.merged_df,
                self.interested_meta_cols,
                postfix=f"{drug}{drug_count}-{vis_mode}-{timestamp}",
                cmap=cmap, figsize=figsize, vis_mode=vis_mode,
                save_dir=self.save_dir
        )


    def plot_individual_cancer_projection(self, drug):
        """Plot individual projection of each cancer."""
        merged_label_df = self.merged_df[["short_sample_id"] + self.interested_meta_cols]
        plot_individual_cancer_project(
                self.merged_df,
                merged_label_df, self.projection,
                key2group="disease_code",
                key2color="binary_response",
                prefix=f"vis_meta-{drug}-Indiv-Original",
                save_dir=self.save_dir
        )

    def interactive_bokeh_for_dgea(self, drug, timestamp, vis_mode, s2_hoverkeys):
        """Interactive Bokeh plot for DGEA."""
        hover_notion = [(key, self.merged_df[key]) for key in s2_hoverkeys]
        interactive_bokeh_with_select_test_for_DGEA(
                self.projection[:, 0], self.projection[:, 1],
                hover_notions=hover_notion,
                height=500, width=500,
                key2color="binary_response",
                title=f"{drug}", mode=vis_mode,
                table_columns=["x", "y"] + s2_hoverkeys,
                s2_hoverkeys=s2_hoverkeys,
                postfix=f"{drug}-{timestamp}-DGEA-{len(self.projection)}",
                if_color_gradient=True,
                if_multi_separate=True,
                save_dir=self.save_dir, scatter_size=6
        )


    def interactive_plot_with_subplots_on_one_key(self, drug, timestamp):
        """Interactive plot with unified color coding across subplots."""
        hover_notion = [(key, self.merged_df[key]) for key in self.interested_meta_cols]
        interactive_bokeh_multi_subplots_unify_colorcoding(
                self.projection[:, 0], self.projection[:, 1],
                features=self.features_df.values,
                hover_notions=hover_notion,
                height=500, width=500,
                title="Title",
                table_columns=["x", "y"] + self.interested_meta_cols,
                s2_hoverkeys=self.interested_meta_cols,
                key2color="binary_response",
                key2separate="disease_code",
                postfix=f"{drug}-{timestamp}-colorby-response-{len(self.projection)}",
                if_color_gradient=True,
                if_indiv_proj=False,
                save_dir=self.save_dir, scatter_size=6
        )

    def interactive_all_meta_subplots(self, drug, timestamp, vis_mode):
        """Interactive plot with all meta-columns in one subplot."""
        hover_notion = [(key, self.merged_df[key]) for key in self.interested_meta_cols]
        interactive_bokeh_all_meta_in1_subplots(
                self.projection[:, 0], self.projection[:, 1],
                self.merged_df[["short_sample_id"] + self.interested_meta_cols],
                hover_notions=hover_notion,
                table_columns=["x", "y"] + self.interested_meta_cols,
                if_color_gradient=True,
                if_indiv_proj=False,
                s2_hoverkeys=self.interested_meta_cols,
                title=f"{drug}",
                mode=vis_mode,
                postfix=f"{drug}-{timestamp}-{len(self.projection)}",
                save_dir=self.save_dir, scatter_size=6
        )



class GDSCDataProcessor:
    """
    load GDSC data
    """
    def __init__(self, name):
        """
        Initialize the class with the path to the dataset.
        """
        self.name = name
        if name.lower() == "gdsc1":
            data_path = r"../data/GDSC_drug_response/PANCANCER_IC_Tue Jul 25 15_05_03 2023.processed.csv"
        elif name.lower() == "gdsc2":
            data_path = r"../data/GDSC_drug_response/PANCANCER_IC_Tue Jul 25 15_05_30 2023.processed.csv"

        self.data_dir = path.dirname(data_path)

        self.gdsc_df = pd.read_csv(data_path)
        self.gdsc_df["Drug Name"] = self.gdsc_df["Drug Name"].apply(lambda x: x.lower())
        if "processed" not in data_path:

            # to match CCLE names, which are without -
            exception_cell_name_list = self.get_exceptions(self.gdsc_df)

            self.gdsc_df["cell_line_name"] = self.gdsc_df[
                "Cell Line Name"].apply(
                    lambda x: self.standardize_cell_line_name(x, exception_cell_name_list)
            )
            self.gdsc_df["diagnosis"] = self.gdsc_df["TCGA Classification"]

            # self.data["cell_line_name"] = self.data["Cell Line Name"].str.split("-").str.join("")
            self.gdsc_df['Unique Drug Name'] = self.gdsc_df['Drug Name'] + "_" + self.gdsc_df['Dataset Version']

            self.gdsc_df.to_csv(data_path[0:-3]+"processed.csv", index=False)

    def get_shared_drugs(self, other_dataset_drug_counts, min_num_samples=20):
        """
        Get the shared drugs between the GDSC dataset and another dataset based on drug counts.
        :param other_dataset_drug_counts:
        :param min_num_samples:
        :return:
        """
        shared_drugs = set(
            other_dataset_drug_counts[other_dataset_drug_counts['total_count'] > min_num_samples]['drug_name']
        ).intersection(set(self.gdsc_df['Drug Name'].apply(lambda x: x.lower())))
        return shared_drugs

    def categorize_auc_3_class(self, auc_value):
        if auc_value < 0.7:
            return 2
        elif 0.7 <= auc_value < 0.85:
            return 1
        else:
            return 0

    def categorize_IC50_3_class(self, IC50):
        if IC50 < -1:  #  (IC50 < 0.37μM
            return 3
        elif -1 <= IC50 < 0:  #  (0.37μM < IC50 < 1μM )
            return 2
        elif 0 <= IC50 < 2:   # (IC50 between 1μM and 7.39μM)
            return 1
        else:
            return 0 #  (IC50 > 7.39μM )

    def categorize_zscore_3_class(self, Z_score):
        if Z_score < -0.8:  #  (responding)
            return 1
        elif -0.8 <= Z_score < 0.8:  #  (partical responding)
            return 0.5
        else:
            return 0 #  (Not responding )

    def categorize_auc_5_class(self, auc_value):
        """
        class, bigger the better
        :param auc_value:
        :return:
        """
        if auc_value < 0.6:
            return 4
        elif 0.6 <= auc_value < 0.7:
            return 3
        elif 0.7 <= auc_value < 0.8:
            return 2
        elif 0.8 <= auc_value < 0.9:
            return 1
        else:
            return 0

    def get_top_k_most_variant_genes(self, expression_data, top_k=1000):
        """
        Get the top K most variant genes based on variance across samples.

        Parameters:
        - expression_data (pd.DataFrame): Gene expression data with genes as columns and samples as rows.
        - top_k (int): Number of top variant genes to retrieve.

        Returns:
        - pd.DataFrame: A DataFrame containing the top K most variant genes.
        """
        # Calculate variance for each gene
        gene_variances = expression_data.var(axis=0)

        # Sort genes by variance in descending order
        top_genes = gene_variances.sort_values(ascending=False).head(top_k)

        # Extract the top K genes and their variance
        top_genes_df = expression_data[top_genes.index]

        return top_genes_df

    def standardize_cell_line_name(self, cell_line_name, exceptions):
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

    def visualize_stats(self):
        """
        Visualize statistics of the dataset:
        - Number of drugs in each cell line name.
        - Number of different TCGA classifications for each drug.
        """
        # Number of drugs in each cell line
        cell_line_stats = self.gdsc_df.groupby('Cell Line Name')['Unique Drug Name'].nunique()
        plt.figure(figsize=(10, 5))
        cell_line_stats.plot(kind='bar')
        plt.title('Number of Drugs Tested in Each Cell Line')
        plt.xlabel('Cell Line Name')
        plt.ylabel('Number of Drugs')
        plt.xticks(rotation=90)
        plt.show()

    def visualize_cancer_types_per_drug(self):
        # Number of TCGA classifications for each drug
        from matplotlib.colors import ListedColormap
        # Create a stacked bar plot for the number of different TCGA classifications for each drug
        drug_tcga_stats = self.gdsc_df.groupby(['Unique Drug Name', 'TCGA Classification']).size().unstack(
            fill_value=0)

        # Generate a palette with 40+ distinct colors using a continuous colormap
        num_colors = drug_tcga_stats.shape[1]  # Specify the number of discrete colors needed
        colors = generate_pastel_colors(num_colors)
        large_palette = ListedColormap(colors)

        # Plot the stacked bar chart
        plt.figure(figsize=(25, 7))
        drug_tcga_stats.plot(kind='bar', stacked=True, figsize=(25, 7), colormap=large_palette)

        # Customize the plot
        plt.title(
            'Distribution of cancer types for each drug', fontsize=14)
        plt.xlabel('Drug Name', fontsize=12)
        plt.ylabel('Count of TCGA Classifications', fontsize=12)
        plt.xticks(rotation=90, fontsize=10)
        plt.legend(
            title='TCGA Classification', fontsize=12, bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def get_exceptions(self, dataframe):
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
        df['standardized_name'] = df['Cell Line Name'].str.replace('-', '').str.upper()
        # Group by the standardized names
        grouped = df.groupby('standardized_name')
        # Initialize a set to hold exceptions
        exception_list = []
        # Iterate over the groups
        for name, group in grouped:
            # If the group contains more than one unique original cell line name with different COSMIC IDs
            if group['Cell Line Name'].nunique() > 1 and group['Cosmic ID'].nunique() > 1:
                # Add all original cell line names in this group to the exceptions set
                exception_list += list(group['Cell Line Name'].unique())
        print(f"Exception name list: {exception_list}")
        return exception_list

    def prepare_drug_grouped_data_for_regressor(self, gex_all_tcga_ccle, meta_all_tcga_ccle):
        """
        For each drug in gdsc_df, prepare a dataset of gene expression data (from gex_all)
        and drug response (from gdsc_df) for training a regressor.
        for one drug
        1. find shared cell lines between gdsc and meta_all(ccle/ccle+tcga)
        2. extract the gene expression of these cell lines
        3. train in leave-drug-out/leave-cell-line-out/random-split/leave-both-out scheme for DRP

        Returns:
            dict: A dictionary where each drug is a key, and the value is a tuple (X, y),
                  with X being the gene expression data and y being the drug response.
        """
        drug_data_dict = {}
        # meta_all_tcga["cell_line_name"] = meta_all_tcga["stripped_cell_line_name"]
        if "cell_line_name" not in gex_all_tcga_ccle.columns:
            gex_all_tcga_ccle.insert(
                0, "cell_line_name", meta_all_tcga_ccle["stripped_cell_line_name"])

        gdsc_only_gex = gex_all_tcga_ccle[meta_all_tcga_ccle["dataset_name"] == "GDSC"]
        gdsc_only_meta = meta_all_tcga_ccle[meta_all_tcga_ccle["dataset_name"] == "GDSC"]
        # Iterate over each unique drug in gdsc_df
        for drug_id in self.gdsc_df["Drug ID"].unique():
            drug = self.gdsc_df[self.gdsc_df["Drug ID"] == drug_id]["Drug Name"].values[0]
            # Filter gdsc_df for the current drug
            gex_X, meta_y = self.gdsc_extract_data_for_one_drug(
                drug_id, gdsc_only_gex, gdsc_only_meta, drug_identifier_col="Drug ID")

            # Store the data in the dictionary
            drug_data_dict[drug_id] = (drug, gex_X, meta_y)

        return drug_data_dict

    def gdsc_extract_data_for_one_drug(self, drug_id, gex_only_gdsc, meta_only_gdsc, drug_identifier_col="Drug Name"):
        gdsc_drug_df = self.gdsc_df[self.gdsc_df[drug_identifier_col] == drug_id]

        # Filter datasets based on 'cell_line_name' and 'TCGA_classification'
        common_cell_lines = set(gdsc_drug_df["cell_line_name"]).intersection(set(meta_only_gdsc["cell_line_name"]))

        # Optionally, convert to a list if needed
        shared_cell_line_names_list = list(common_cell_lines)

        # Filter both gdsc_df and gex_all to keep only the common cell lines
        filtered_gdsc_meta = gdsc_drug_df[
            gdsc_drug_df["cell_line_name"].isin(shared_cell_line_names_list)].groupby(
            "cell_line_name").first().reset_index()  #
        filtered_gex = gex_only_gdsc[
            gex_only_gdsc["cell_line_name"].isin(shared_cell_line_names_list)].groupby(
            "cell_line_name").first().reset_index()

        # Align the dataframes by cell line name
        filtered_gdsc_meta.index = filtered_gdsc_meta["cell_line_name"]

        filtered_gex = filtered_gex.set_index("cell_line_name")

        # Ensure the cell lines are aligned
        filtered_gex = filtered_gex.loc[filtered_gdsc_meta.index]
        assert len(filtered_gdsc_meta) == len(
            filtered_gex), "numbers of filtered_gex and filtered_gdsc not matching!"
        # Prepare the input (gene expression) and output (drug response)
        X = filtered_gex.drop(columns=["cell_line_name"], errors="ignore")
        y = filtered_gdsc_meta  # Replace "drug_response" with the actual column name
        return X, y

    def load_gdsc_drug_grouped_data(self, gex_all, meta_all, save_prefix="GDSC",
                                load_from_saved_filename=None):
        """
        Perform a leave-drug-out train-validation split on the data prepared for regressors.

        Returns:
            dict: A dictionary with 'train' and 'val' keys for each drug.
        :param gex_all:
        :param meta_all:
        :param split_case: four cases: random-split, leave-drug-out, eave-cell-line-out, leave-both-out
        :param test_size:
        :param random_state:
        :param save_prefix:
        :param load_from_saved_filename:
        :return:
        """

        if gex_all.shape[1] > 19000: ## original data
            gex_data_filtered = self.get_top_k_most_variant_genes(gex_all, top_k=2500)
        else:
            gex_data_filtered = gex_all.copy()

        if not load_from_saved_filename:
            drug_data_dict = self.prepare_drug_grouped_data_for_regressor(gex_data_filtered, meta_all)

            drug_grouped_data_dict = {}
            with open(path.join(self.data_dir, f'{save_prefix}_{len(drug_data_dict)}drugs.pickle'), 'wb') as handle:
                drug_grouped_data_dict["data"] = drug_data_dict
                pickle.dump(drug_grouped_data_dict, handle)

        else:  # the load_from_saved_filename is given
            if path.isfile(load_from_saved_filename):
                with open(load_from_saved_filename, "rb") as file:
                    loaded_data = pickle.load(file)
                    drug_data_dict = loaded_data["data"]
            else:
                print("Given file is not a file!")
        return drug_data_dict

    def filter_and_align(self, data_df, cell_lines, gex_df):
        """Filters and aligns the gene expression and drug response data."""
        filtered_data = data_df[data_df["cell_line_name"].isin(cell_lines)]
        filtered_gex = gex_df[gex_df["cell_line_name"].isin(cell_lines)]
        filtered_gex = filtered_gex[~filtered_gex["cell_line_name"].duplicated(keep='first')]  # there might be multiple version with the same cell line name

        # Align the dataframes by cell line name
        filtered_data = filtered_data.set_index("cell_line_name")
        filtered_gex = filtered_gex.set_index("cell_line_name")  # OCIAML5 has different version from GDSC

        # Ensure alignment
        filtered_gex = filtered_gex.loc[filtered_data.index]
        return filtered_gex.drop(columns=["cell_line_name"], errors="ignore"), filtered_data


    def prepare_random_split(self, gex_all_df, meta_all_df,
                                                     test_fraction=0.2,
                                                     random_seed=42):
        """
        For each drug in gdsc_df, prepare a dataset of gene expression data (from gex_all)
        and drug response (from gdsc_df) for training a regressor in a leave-cell-line-out scheme.

        Args:
            gex_all_df (DataFrame): Gene expression data including cell line information.
            meta_all_df (DataFrame): Metadata containing cell line names and diagnoses.
            y_col (str): The column name for drug response in gdsc_df.
            test_fraction (float): Fraction of cell lines to leave out for testing.
            random_seed (int): Seed for reproducibility.

        Returns:
            dict: A dictionary where each drug is a key, and the value is a tuple (X_train, y_train, X_test, y_test),
                  with X being the gene expression data and y being the drug response.
        """

        random.seed(random_seed)
        drug_data_dict = {}

        # Add 'cell_line_name' to gex_all_tcga_ccle if not already present
        if "cell_line_name" not in gex_all_df.columns:
            gex_all_df.insert(
                    0, "cell_line_name", meta_all_df["stripped_cell_line_name"]
            )
        gex_gdsc_df = gex_all_df[meta_all_df["dataset_name"] == "GDSC"]
        meta_gdsc_df = meta_all_df[meta_all_df["dataset_name"] == "GDSC"]

        # Iterate over each unique drug in gdsc_df
        uniq_drugs = self.gdsc_df["Drug ID"].unique()
        for drug_id in uniq_drugs:
            # Filter data for the specific drug_id
            drug_subset = self.gdsc_df[self.gdsc_df['Drug ID'] == drug_id]

            drug_name = drug_subset[drug_subset["Drug ID"] == drug_id]["Drug Name"].values[0]

            # ## get cell lines that with gene expression available
            shared_cell_line_names = set(
                    drug_subset["cell_line_name"]
                    ).intersection(
                    set(meta_gdsc_df["cell_line_name"])
            )

            X_train_test, y_train_test_df = self.filter_and_align(drug_subset, shared_cell_line_names, gex_gdsc_df)

            # Save to dictionary
            drug_data_dict[drug_id] = (
                    drug_name,
                    X_train_test.astype(np.float16),
                    y_train_test_df
            )

        return drug_data_dict

def save_drug_data_dict_to_pickle(drug_data_dict, prefix="prefix", save_dir="../data/GDSC_drug_response"):
    train_dict = {}
    val_dict = {}
    # Loop through drug_data_dict
    for drug_id, data_tuple in drug_data_dict.items():
        drug_name = data_tuple[0]  # Extract the drug name
        train_x_df = data_tuple[1]  # Training features
        train_y_df = data_tuple[2]  # Training targets
        val_x_df = data_tuple[3]  # Validation features
        val_y_df = data_tuple[4]  # Validation targets

        # Parse training data
        train_dict[drug_id] = (
                drug_name,
                train_x_df.astype(np.float16),
                train_y_df,
        )

        # Parse validation data
        val_dict[drug_id] = (
                drug_name,
                val_x_df.astype(np.float16),
                val_y_df,
        )

    data_dict = {}
    with open(
            f"{save_dir}/{prefix}_L-Cell-O_train{len(drug_data_dict)}_val{len(drug_data_dict)}_randxxx.pickle",
            'wb') as handle:
        data_dict["train"] = train_dict
        data_dict["val"] = val_dict
        pickle.dump(data_dict, handle)

def save_drug_data_dict_to_dataframe(drug_data_dict, prefix="prefix", save_dir="../data/GDSC_drug_response"):
    train_rows = []
    test_rows = []

    # Loop through the dictionary
    count = 0
    for drug_id, data_tuple in drug_data_dict.items():
        if count < len(drug_data_dict):
            drug_name = data_tuple[0]  # Extract drug name

            # Process train_x and train_y
            train_x_df = pd.DataFrame(data_tuple[1]).astype(np.float16)  # Features for training
            train_y_df = pd.DataFrame(data_tuple[2])  # Targets for training
            train_y_df = train_y_df.astype(
                    {col: np.float16 for col in
                     train_y_df.select_dtypes(include=[np.float64]).columns})

            # Add metadata columns for train
            train_x_df.insert(0, "train_x_drug_id", drug_id)
            train_x_df.insert(0, "train_x_drug_name", drug_name)
            train_x_df.insert(0, "split", "train")
            train_x_df.insert(0, "train_x_cell_line_name", train_x_df.index)
            train_y_df.insert(0, "train_y_drug_id", drug_id)
            train_y_df.insert(0, "train_y_drug_name_y", drug_name)
            train_y_df.insert(0, "train_y_split", "train")

            # Combine train_x and train_y into a single DataFrame (row-wise for the same split)
            train_combined = pd.concat(
                    [train_y_df, train_x_df], axis=1)
            train_rows.append(train_combined)

            # Process test_x and test_y
            test_x_df = pd.DataFrame(data_tuple[3]).astype(np.float16)  # Features for testing
            test_y_df = pd.DataFrame(data_tuple[4])  # Targets for testing
            test_y_df = test_y_df.astype(
                    {col: np.float16 for col in
                     test_y_df.select_dtypes(include=[np.float64]).columns})

            # Add metadata columns for test
            test_x_df.insert(0, "test_x_drug_id", drug_id)
            test_x_df.insert(0, "test_x_drug_name", drug_name)
            test_x_df.insert(0, "split", "test")
            test_x_df.insert(0, "test_x_cell_line_name", test_x_df.index)
            test_y_df.insert(0, "test_y_drug_id", drug_id)
            test_y_df.insert(0, "test_y_drug_name_y", drug_name)
            test_y_df.insert(0, "split", "test")

            # Combine test_x and test_y into a single DataFrame (row-wise for the same split)
            test_combined = pd.concat(
                    [test_y_df, test_x_df], axis=1)
            # test_combined = pd.concat(
            #         [test_y_df.reset_index(drop=True), test_x_df.reset_index(drop=True)], axis=1)
            test_rows.append(test_combined)
        count += 1

    # Concatenate all rows into a single DataFrame
    train_big_df = pd.concat(train_rows)
    test_big_df = pd.concat(test_rows)
    # train_big_df = pd.concat(train_rows, ignore_index=True)
    # test_big_df = pd.concat(test_rows, ignore_index=True)

    train_big_df.to_csv(
        path.join(save_dir, f"{prefix}_L_Cellline_O_train_{len(train_rows)}drugs.csv"),
        index=False)
    test_big_df.to_csv(
        path.join(save_dir, f"{prefix}_L_Cellline_O_test_{len(train_rows)}drugs.csv"),
        index=False)

def categorize_zscore_6_class(zscore):
    if zscore < -1:  #  (IC50 < 0.37μM
        return 5
    elif -1 <= zscore < -0.5:  #  (0.37μM < IC50 < 1μM )
        return 4
    elif 0 <= zscore < 0.3:  #  (0.37μM < IC50 < 1μM )
        return 3
    elif 0.3 <= zscore < 0.6:   # (IC50 between 1μM and 7.39μM)
        return 2
    elif 0.6 <= zscore < 1:   # (IC50 between 1μM and 7.39μM)
        return 1
    else:
        return 0 #  (IC50 > 7.39μM )

class GDSCModelCollection:
    def __init__(self, gexmix, model_base_dir="models"):
        """
        Initialize the GDSCModelCollection for training and inference.

        Args:
            gdsc_processor (GDSCDataProcessor): An instance of the GDSC data processor.
            model_dir (str): Directory to save and load models.
        """
        self.gexmix = gexmix
        self.model_base_dir = model_base_dir

        makedirs(self.model_base_dir, exist_ok=True)

    def augment_data(self, X, target_label_dict,
                     aug_by_column="TCGA Classification",
                     num2aug=100,
                     keys4mix=["tcga_hot_labels", "tumor_percent", "normal_percent"],
                     if_include_original=False):
        """
        Apply data augmentation using the GExMix instance.

        Args:
            X (pd.DataFrame): Original gene expression data.
            y (np.ndarray): Target labels.
            aug_by_column (str): Column to stratify augmentation by.
            target_label_dict (dict): Label dictionary.
            num2aug (int): Number of augmented samples.
            keys4mix (list): Keys for mixing.

        Returns:
            tuple: Augmented gene expression data and labels.
        """
        if self.gexmix is None:
            raise ValueError("GExMix instance is required for data augmentation.")

        return self.gexmix.augment_random(
                aug_by_column=aug_by_column,
                target_features=X,
                target_label_dict=target_label_dict,
                if_stratify=False,
                num2aug=num2aug,
                random_state=99,
                keys4mix=keys4mix,
                if_include_original=if_include_original,
                save_dir=self.model_base_dir)


    def fit_model(self, model, model_type, X_train, y_train, **kwargs):
        """
                Handles training for different types of models.
        Args:
            model: The model to train (e.g., Keras, sklearn, etc.).
            model_type: A string indicating the type of model ('keras', 'sklearn', etc.).
            X_train: Training data features.
            y_train: Training data labels.
            kwargs: Additional parameters like epochs, batch_size for Keras models.
        Returns:
            The trained model.
        :param model:
        :param model_type:
        :param X_train:
        :param y_train:
        :param kwargs:
        :return:
        """

        if model_type == 'keras':
            from tensorflow.keras.callbacks import EarlyStopping
            # EarlyStopping Callback
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True)
            epochs = kwargs.get('epochs', 50)
            batch_size = kwargs.get('batch_size', 32)
            val_x = kwargs.get('val_x')
            val_y = kwargs.get('val_y')
            x_val = tf.convert_to_tensor(val_x)
            y_val = tf.convert_to_tensor(val_y)
            history = model.fit(
                    X_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=[x_val, y_val], callbacks=early_stopping)
        elif model_type == 'sklearn':
            model.fit(X_train, y_train)
            history = None
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model, history

    def save_model(self, model, model_type, model_path):
        if model_type == 'keras':
            model.save(model_path.replace(".pkl", ".h5"))
        elif model_type == 'sklearn':
            joblib.dump(model, model_path)

    def format_drug_name(self, drug_name):
        return drug_name.replace("/", "+")

    def plot_overall_performance(self, performance_metrics, prefix="", save_dir="./"):
        """
        Plot overall performance metrics for all drugs.

        Args:
            performance_metrics (dict): Dictionary containing performance metrics for all drugs.
        """
        # metrics = [ele for ele in performance_metrics.get("ori", {}).keys() if "score" in ele or "rho" in ele]
        # ori_metrics = performance_metrics.get("ori", {})
        # aug_metrics = performance_metrics.get("aug", {})
        #
        # fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
        # ori_or_aug = ""
        # for i, metric in enumerate(metrics):
        #     ori_values = [item[2] for item in ori_metrics.get(metric)]
        #     aug_values = [item[2] for item in aug_metrics.get(metric)]
        #     drugs = [item[0] for item in aug_metrics.get(metric)]
        #     x = np.arange(max(len(aug_values), len(ori_values)))
        #     width = 0.35
        #     if len(ori_values) > 0:
        #         axes[i].bar(x - width / 2, ori_values, width, label='Original')
        #         ori_or_aug += "+ori"
        #         prefix += f"{len(ori_values)}"
        #     if len(aug_values) > 0:
        #         axes[i].bar(x + width / 2, aug_values, width, label='Augmented')
        #         ori_or_aug += "+aug"
        #     axes[i].set_xlabel('Drugs' if i == (len(metric) - 1) else "")
        #     axes[i].set_ylabel(metric)
        #     axes[i].set_title(f'{metric} for all drugs')
        #     # set rotation for x-axis labels
        #     axes[i].set_xticks(x, drugs, rotation=45, ha='right')
        #     axes[i].legend(fontsize=10, bbox_to_anchor=(1., 1), loc='upper left')
        # plt.tight_layout()

        # Assume performance_metrics contains original and augmented metrics
        metrics = [ele for ele in performance_metrics.get("ori", {}).keys() if
                   "score" in ele or "rho" in ele]
        ori_metrics = performance_metrics.get("ori", {})
        aug_metrics = performance_metrics.get("aug", {})

        fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 8), sharex=True)
        ori_or_aug = ""
        for i, metric in enumerate(metrics):
            # Get the original and augmented metrics
            ori_values_raw = ori_metrics.get(metric, [])
            aug_values_raw = aug_metrics.get(metric, [])

            # Extract drugs, values, and calculate mean and std for error bars
            ori_drugs = [item[0] for item in ori_values_raw]
            aug_drugs = [item[0] for item in aug_values_raw]

            # Group values by drug for averaging and std deviation
            unique_drugs = list(set(ori_drugs + aug_drugs))
            ori_means, ori_stds, aug_means, aug_stds = [], [], [], []

            for drug in unique_drugs:
                # Original
                ori_values = [item[2] for item in ori_values_raw if item[0] == drug]
                if ori_values:
                    ori_means.append(np.mean(ori_values))
                    ori_stds.append(np.std(ori_values))
                else:
                    ori_means.append(0)  # Default value if no data
                    ori_stds.append(0)

                # Augmented
                aug_values = [item[2] for item in aug_values_raw if item[0] == drug]
                if aug_values:
                    aug_means.append(np.mean(aug_values))
                    aug_stds.append(np.std(aug_values))
                else:
                    aug_means.append(0)  # Default value if no data
                    aug_stds.append(0)

            # Bar plot with error bars
            x = np.arange(len(unique_drugs))
            width = 0.35
            axes[i].bar(x - width / 2, ori_means, width, yerr=ori_stds, label='Original', capsize=5)
            axes[i].bar(
                x + width / 2, aug_means, width, yerr=aug_stds, label='Augmented', capsize=5)

            # Set titles, labels, and ticks
            axes[i].set_ylabel(metric)
            axes[i].set_title(f'{metric} for all drugs')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(unique_drugs, rotation=45, ha='right')
            axes[i].legend(fontsize=10, bbox_to_anchor=(1., 1), loc='upper left')
        plt.savefig(path.join(self.model_base_dir, f"{prefix} drugs_overall_performance.png"))
        plt.close()

    def plot_history_train_val(self, history_dict, title="title", prefix="run_postfix",
                               save_dir="save_dir"):
        # Plot each key in a separate subplot
        keys = list(history_dict.keys())
        num_metrics = len(keys) // 2
        fig, axs = plt.subplots(num_metrics, 1, figsize=(10, num_metrics * 4))
        plt.suptitle(title)
        for i, key in enumerate([k for k in keys if 'val_' not in k]):
            axs[i].plot(history_dict[key], label=f'Training {key}')
            axs[i].plot(history_dict[f'val_{key}'], label=f'Validation {key}')
            axs[i].set_title(f'{key} over Epochs')
            axs[i].set_ylabel(key)
            axs[i].set_xlabel('Epoch')
            axs[i].legend(loc='upper left')
            axs[i].grid(True)
        plt.tight_layout()
        plt.savefig(path.join(save_dir, f"{prefix}-learning_curve.png"))
        plt.close()

    def train_and_save_one_model(self, model, model_type, drug_key:int,
                                 drug: str, X_train: pd.DataFrame,
                                 y_train: np.ndarray, model_name: str,
                                 beta=0, gene_start_col=0,
                                 val_data_list=None, y_col="IC50", postfix="",
                                 ori_or_aug="ori"):
        """
        Train and save a single model for a specific drug.

        Args:
            key (int): Drug identifier.
            drug (str): Drug name.
            X_train (np.ndarray): Input features.
            y_train (np.ndarray): Target labels.
            model_name (str): Name of the regression model.
            ori_or_aug (str): Indicates whether training is on original or augmented data.
        """
        aug_kwargs = self.get_fit_aug_args(model_type, val_data_list[1], val_data_list[2][y_col])

        model, history = self.fit_model(model, model_type, X_train, y_train, **aug_kwargs)
        if model_type == 'keras':
            # if self.coll_history is empty, then collect the history
            if not self.coll_history:
                for key in history.history.keys():
                    self.coll_history[key] = []
            for key in history.history.keys():
                self.coll_history[key].extend(history.history[key])

            self.plot_history_train_val(self.coll_history, title=f"{drug.capitalize()} {model_name} Training",
                                        prefix=f"{drug}_{model_name}_{postfix}",
                                        save_dir=self.model_base_dir)

        drug = self.format_drug_name(drug)  # repalce /

        model_path = path.join(
            self.model_base_dir, f"{ori_or_aug}_{model_name}_No{drug_key}_drug{drug}.pkl")

        self.save_model(model, model_type, model_path)

        if val_data_list is not None:
            val_drug, X_val, y_val = val_data_list

            val_drug = self.format_drug_name(val_drug)
            val_prediction = model.predict(X_val.iloc[:, gene_start_col:])
            # If prediction is 2D, take the first column
            if val_prediction.ndim > 1:
                if val_prediction.shape[1] > 1:
                    val_prediction = val_prediction[:, 1]
                else:
                    val_prediction = val_prediction.flatten()
            # Combine ground truth (y_val["IC50"]) and predictions into a DataFrame
            gt_predict_df = pd.DataFrame(
                    {
                            "ground_truth": y_val[y_col].reset_index(drop=True),  #
                            "prediction": val_prediction
                    })
            gt_predict_df.index = y_val.index

            # Calculate evaluation metrics
            pearson_corr, _ = pearsonr(y_val[y_col], val_prediction)
            sp_corr, p_value = spearmanr(y_val[y_col], val_prediction)
            r2 = r2_score(y_val[y_col], val_prediction)
            merged_meta_w_prediction = pd.concat([y_val, gt_predict_df], axis=1)
            corr_str = f"Pearson r: {pearson_corr:.2f}\nSpearman r: {sp_corr:.2f}\nR2:{r2:.2f}"

            # Store performance metrics
            self.performance_metrics[ori_or_aug]["pearson_score"].append([drug, beta, pearson_corr])
            self.performance_metrics[ori_or_aug]["spearman_score"].append([drug, beta, sp_corr])
            self.performance_metrics[ori_or_aug]["r2_score"].append([drug, beta, r2])
            self.performance_metrics[ori_or_aug]["pred_w_ori_meta"].append([drug, beta, merged_meta_w_prediction])

            plt.figure(figsize=(8, 6))
            for jj, (key, value) in enumerate(self.performance_metrics["ori"].items()):
                if "score" in key and len(value) > 0:
                    value = np.array(value[0])
                    label = "Original" if jj == 0 else ""
                    plt.scatter(value[1].astype(np.float32), value[2].astype(np.float32), marker="*", label=label)
            for key, value in self.performance_metrics['aug'].items():
                if "score" in key and len(value) > 0:
                    value = np.array(value)
                    plt.plot(value[:, 1].astype(np.float32), value[:, 2].astype(np.float32), label=key)
                    plt.legend()
            plt.title(f"{drug.capitalize()} {model_name} Aug. Performance")
            plt.xlabel("Beta")
            plt.ylabel("Score")
            plt.tight_layout()
            plt.savefig(path.join(self.model_base_dir, f"{drug}_{model_name}_aug_performance.png"))
            plt.close()

            self.visualize_prediction_gt(
                gt_predict_df, y_val, y_col=y_col, prefix=f"{val_drug.capitalize()}-{ori_or_aug.capitalize()}-{postfix}",
                    corr_str=corr_str, save_dir=self.model_base_dir)
            self.visualize_prediction_gt_ranks(gt_predict_df, y_val,
                                               y_col="IC50", prefix=f"{val_drug.capitalize()}-{ori_or_aug.capitalize()}-rank-{postfix}",
                                               corr_str=corr_str,
                                               save_dir=self.model_base_dir)

            return y_val[y_col], val_prediction
        print(f"Model for No.{drug_key} drug {drug} saved at {model_path}")

    def visualize_prediction_gt(self, gt_predict_df, y_df, y_col="IC50", corr_str="str", prefix="prefix", save_dir="./"):
        """
        Visualizes ground truth and prediction comparisons.

        Args:
            gt_predict_df (pd.DataFrame): DataFrame with ground truth and predictions.
            y_df (pd.DataFrame) with groundtruth: DataFrame with additional metadata, e.g., TCGA Classification.
            prefix (str): Prefix for plot titles and saved file names.
            save_dir (str): Directory to save plots.
        """
        # Extract ground truth and prediction values
        ground_truth = gt_predict_df.iloc[:, 0]
        prediction = gt_predict_df.iloc[:, -1]
        gt_predict_df.index = y_df.index

        # Plot line comparison of ground truth vs predictions
        # self._plot_gt_pred_line_comparison(
        #         ground_truth, prediction, gt_predict_df.index,
        #         prefix, corr_str=corr_str, y_col=y_col, save_dir=save_dir
        # )

        # Plot scatter comparison of ground truth vs predictions
        self._plot_gt_pred_scatter_comparison(
                ground_truth, prediction, prefix, corr_str=corr_str, y_col=y_col, save_dir=save_dir
        )

        # Create a split violin plot grouped by TCGA classification
        # self._plot_gt_pred_violin_grouped_by_cancer(
        #         gt_predict_df, y_df, prefix, corr_str=corr_str, y_col=y_col, save_dir=save_dir
        # )
        self._plot_violin_grouped_by_metric(
                data_df=pd.concat([y_df, gt_predict_df], axis=1),
                group_by_col="TCGA Classification",
                metric1_col="ground_truth",
                metric2_col="prediction",
                metric1_label="Ground Truth",
                metric2_label="Prediction",
                order_by_mean_col="Ground Truth",
                prefix=prefix.capitalize(),
                corr_str=corr_str,
                save_dir=save_dir,
                ylabel=y_col.capitalize(),
                plot_mode="box"
        )

    def visualize_prediction_gt_ranks(self, gt_predict_df, y_df, y_col="IC50", prefix="prefix",
                                      corr_str="corr_str",
                                      save_dir="./"):
        """
        Visualizes ground truth vs predictions with rank-based metrics:
        - Scatter plot of ranks
        - Rank difference histogram
        - Cumulative rank error plot

        Args:
            gt_predict_df (pd.DataFrame): DataFrame containing predictions.
            y_df (pd.DataFrame): DataFrame containing ground truth values.
            y_col (str): Column name for ground truth values in y_df.
            prefix (str): Prefix for plot titles and saved file names.
            save_dir (str): Directory to save plots.
        """
        # Perfect predictor model: predictions equal to the ground truth
        gt_predict_df.index = y_df.index
        predictions_df = pd.concat([y_df, gt_predict_df], axis=1)

        # Rank-based metrics preparation
        predictions_df['ground_truth_rank'] = predictions_df[y_col].rank()
        predictions_df['prediction_rank'] = predictions_df['prediction'].rank()
        predictions_df['rank_difference'] = predictions_df['ground_truth_rank'] - predictions_df[
            'prediction_rank']
        predictions_df['abs_rank_error'] = predictions_df['rank_difference'].abs()

        # Generate rank-related plots
        self._plot_rank_difference_histogram(
            predictions_df, prefix, corr_str=corr_str, save_dir=save_dir)
        self._plot_rank_difference(predictions_df, prefix, corr_str=corr_str, save_dir=save_dir)

        self._plot_violin_grouped_by_metric(
                data_df=predictions_df,
                group_by_col="TCGA Classification",
                metric1_col="ground_truth_rank",
                metric2_col="prediction_rank",
                metric1_label="Ground Truth Rank",
                metric2_label="Prediction Rank",
                order_by_mean_col="Ground Truth Rank",
                prefix=prefix.capitalize(),
                corr_str=corr_str,
                ylabel=y_col.capitalize() + " Rank",
                save_dir=save_dir,
                plot_mode="box")


    def _plot_rank_difference(self, predictions_df, prefix, group_by_col='TCGA Classification', corr_str="corr_str", save_dir="./"):
        predictions_df['rank_difference'] = predictions_df['ground_truth_rank'] - predictions_df[
            'prediction_rank']
        sort_df = predictions_df.sort_values("ground_truth_rank")
        sample_counts = predictions_df[group_by_col].astype(str).value_counts()

        predictions_df[f'{group_by_col} (count)'] = predictions_df[group_by_col].map(
                lambda x: f"{x} (n={sample_counts[str(x)]})"  # there are nan classes
        )
        # Group by TCGA Classification and calculate mean ground truth
        grouped = predictions_df.groupby(group_by_col).apply(
                lambda x: x.assign(mean_ground_truth=x['ground_truth_rank'].mean())
        )

        # Sort by mean ground truth
        sorted_grouped = grouped.sort_values('mean_ground_truth')

        # Plot rank differences for each group
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            x=f'{group_by_col} (count)', y='rank_difference',
            data=sorted_grouped, showfliers=False)
        sns.scatterplot(
                x=f'{group_by_col} (count)', y='mean_ground_truth',
                data=sorted_grouped)

        plt.axhline(0, color="red", linestyle="--", label="Perfect Rank Alignment")
        plt.title(f"{prefix.capitalize()}\nRank Differences (Ground Truth - Prediction)")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.xlabel("Sample Index")
        plt.ylabel("Rank Difference")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{prefix}_sorted_rank_difference.png")
        plt.close()

    def _plot_rank_scatter(self, predictions_df, prefix, x_col="ground_truth",
                           y_col="prediction", corr_str="corr_str", save_dir="./"):
        """
        Creates a scatter plot of ground truth ranks vs prediction ranks.
        """
        plt.figure(figsize=(8, 6))

        plt.scatter(
                predictions_df[x_col],
                predictions_df[y_col],
                alpha=0.7, label="Rank Data Points", color="blue"
        )
        plt.plot(
                [predictions_df[x_col].min(),
                 predictions_df[x_col].max()],
                [predictions_df[x_col].min(),
                 predictions_df[x_col].max()],
                color="gray", linestyle="--", label="Optimal Alignment"
        )
        plt.text(
                1.03, 0.95, corr_str,
                transform=plt.gca().transAxes, fontsize=15, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
        )
        plt.title("\n".join(wrap(f"{prefix} - {x_col.capitalize()} vs {y_col.capitalize()}", 45)))
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend(bbox_to_anchor=(1.02, 0.5), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{prefix}_scatter_{x_col}.png")
        plt.close()

    def _plot_rank_difference_histogram(self, predictions_df, prefix, corr_str="corr_str", save_dir="./"):
        """
        Creates a histogram of rank differences.
        """
        plt.figure(figsize=(8, 6))
        plt.hist(predictions_df['rank_difference'], bins=20, color="purple", alpha=0.7)
        plt.text(
                1.03, 0.95, corr_str,
                transform=plt.gca().transAxes, fontsize=15, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
        )
        plt.legend(bbox_to_anchor=(1.02, 0.5), loc='upper left')
        plt.title("\n".join(wrap(f"{prefix} - Distribution of Rank Differences", 60)))
        plt.xlabel("Rank Difference (Ground Truth - Prediction)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{prefix}_rank_difference_histogram.png")
        plt.close()

    def _plot_cumulative_rank_error(self, predictions_df, prefix, corr_str="corr_str", save_dir="./"):
        """
        Creates a cumulative rank error plot.
        """
        sorted_errors = predictions_df['abs_rank_error'].sort_values().cumsum()
        plt.figure(figsize=(8, 6))
        plt.plot(sorted_errors.index, sorted_errors.values, color="blue")
        plt.title("\n".join(wrap(f"{prefix} - Cumulative Rank Error", 60)))
        plt.xlabel("Samples (sorted by absolute rank error)")
        plt.ylabel("Cumulative Absolute Rank Error")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{prefix}_cumulative_rank_error.png")
        plt.close()

    def _plot_gt_pred_line_comparison(self, ground_truth, prediction, cell_lines, prefix, corr_str="corr-p",
                                      y_col="IC50",
                                      save_dir="./"):
        """
        Plots line comparison of ground truth vs predictions.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(cell_lines, ground_truth, label="Ground Truth", marker="o")
        plt.plot(cell_lines, prediction, label="Predicted", marker="x")
        plt.text(
                1.03, 1, corr_str,
                transform=plt.gca().transAxes, fontsize=15, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
        )
        plt.xlabel("Cell Line")
        plt.ylabel(f"{y_col} Value")
        plt.title("\n".join(wrap(f"{prefix.capitalize()} Ground Truth vs Predicted {y_col}", 60)))
        plt.xticks(rotation=45, ha="right")
        plt.legend(bbox_to_anchor=(1.02, 0.5), loc='upper left')
        plt.tight_layout()
        plt.savefig(path.join(save_dir, f"{prefix}-line-gt-vs-pred-{y_col}.png"))
        plt.close()

    def _plot_gt_pred_scatter_comparison(self, ground_truth, prediction, prefix, corr_str="corr_str",
                                         y_col="IC50",
                                         save_dir="./"):
        """
        Plots scatter comparison of ground truth vs predictions.
        """
        plt.figure(figsize=(10, 6.8))
        plt.scatter(ground_truth, prediction, alpha=0.7, label="Data Points", color="blue")
        min_val = min(ground_truth.min(), prediction.min())
        max_val = max(ground_truth.max(), prediction.max())
        plt.plot(
                [min_val, max_val], [min_val, max_val], color="gray", linestyle="--",
                label="Perfect Prediction"
        )
        plt.text(
                1.03, 0.95, corr_str,
                transform=plt.gca().transAxes, fontsize=15, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
        )
        plt.xlabel(f"Ground Truth {y_col}")
        plt.ylabel(f"Predicted {y_col}")
        plt.title(f"{prefix.capitalize()} \nGround Truth vs Predicted (n={len(ground_truth)})")
        plt.legend(bbox_to_anchor=(1.02, 0.5), loc='upper left')
        plt.tight_layout()
        plt.savefig(path.join(save_dir, f"{prefix}-scatter-gt-vs-pred-{y_col}.png"))
        plt.close()

    def _plot_violin_grouped_by_metric(
            self, data_df, group_by_col, metric1_col, metric2_col, metric1_label, metric2_label,
            order_by_mean_col, ylabel="values", prefix="prefix", corr_str=None, save_dir="./", plot_title=None, plot_mode="box"
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
        # Prepare data for violin plot
        # Prepare data for violin plot
        metric1_values = data_df[[group_by_col, metric1_col]].rename(columns={metric1_col: 'value'})
        metric1_values['value_type'] = metric1_label
        metric2_values = data_df[[group_by_col, metric2_col]].rename(columns={metric2_col: 'value'})
        metric2_values['value_type'] = metric2_label
        # Combine data
        extended_df = pd.concat([metric1_values, metric2_values], ignore_index=True)
        extended_df[group_by_col] = extended_df[group_by_col].astype(str)
        # Group by TCGA Classification and calculate mean for ordering

        # Group by TCGA Classification and calculate mean for ordering
        grouped = extended_df.groupby(group_by_col).apply(
                lambda x: x.assign(mean_col=x.loc[x['value_type'] == order_by_mean_col, 'value'].mean())
        ).reset_index(drop=True)
        sorted_grouped = grouped.sort_values('mean_col')
        # Compute Spearman correlation coefficients
        spearman_results = {}
        for group in sorted_grouped[group_by_col].unique():
            group_data = sorted_grouped[sorted_grouped[group_by_col] == group]
            metric1_group = group_data[group_data['value_type'] == metric1_label]['value']
            metric2_group = group_data[group_data['value_type'] == metric2_label]['value']
            if len(metric1_group) > 1 and len(metric2_group) > 1:  # Ensure sufficient data points
                rho, p_value = spearmanr(metric1_group, metric2_group)
                spearman_results[group] = (rho, p_value)
            else:
                spearman_results[group] = (None, None)
        # Create boxplot with ordered x-axis
        unique_classes = sorted_grouped[group_by_col].unique()
        plt.figure(figsize=(max(len(unique_classes)*0.65, 8), 6))
        for i, cls in enumerate(unique_classes):
            if i % 2 == 0:  # Add alternating gray background shades
                plt.axvspan(i - 0.5, i + 0.5, color='lightgray', alpha=0.3, zorder=0)
        ax = sns.boxplot(
                x=group_by_col, y='value', hue='value_type', data=sorted_grouped,
                palette='pastel',  ## order=unique_classes
        )
        # Customize plot
        plt.legend(loc="lower right")
        # Update the x-tick labels to include sample counts
        group_counts = sorted_grouped.groupby(group_by_col).size() / 2
        # Add correlation text to the top of each violin
        xticks = ax.get_xticks()
        xtick_labels = []
        for i, group in enumerate(sorted_grouped[group_by_col].unique()):
            if i == 23:
                print("ok")
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
        ax.set_xticklabels(xtick_labels, rotation=45)
        plt.text(
                1.03, 1.0, corr_str,
                transform=plt.gca().transAxes, fontsize=15, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
        )
        plt.legend(bbox_to_anchor=(1.02, 0.5), loc='upper left')
        # Set plot title and labels
        plot_title = f'{prefix.capitalize()}\n{metric1_label} vs {metric2_label}\n Grouped by {group_by_col}'
        plt.title(plot_title)
        plt.xlabel(group_by_col)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(path.join(save_dir, f"{prefix}-violin-grouped-by-{group_by_col}.png"))
        plt.close()

    def _plot_gt_pred_violin_grouped_by_cancer(self, gt_predict_df, y_df, prefix,
                                               corr_str="corr_str", group_by_col="TCGA Classification",
                                               y_col="IC50", save_dir="./"):
        """
        Creates a split violin plot grouped by TCGA classification.
        """
        # Merge metadata
        gt_predict_df.index = y_df.index
        merged_df = pd.concat([y_df, gt_predict_df[['ground_truth', 'prediction']]], axis=1)

        # Prepare data for violin plot
        true_values = merged_df[[group_by_col, 'ground_truth']].rename(
            columns={'ground_truth': 'value'})
        true_values['value_type'] = 'True'

        predicted_values = merged_df[[group_by_col, 'prediction']].rename(
            columns={'prediction': 'value'})
        predicted_values['value_type'] = 'Prediction'

        extended_df = pd.concat([true_values, predicted_values], ignore_index=True)

        # Plot violin plot
        unique_classes = extended_df[group_by_col].unique()
        plt.figure(figsize=(max(len(unique_classes) * 0.65, 10), 6))
        sns.violinplot(
                x=group_by_col, y='value', hue='value_type', data=extended_df,
                split=True, palette='pastel'
        )
        plt.text(
                1.03, 0.95, corr_str,
                transform=plt.gca().transAxes, fontsize=15, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
        )
        plt.legend(bbox_to_anchor=(1.02, 0.5), loc='upper left')
        plt.title("\n".join(wrap(f'{prefix.capitalize()} \nGround Truth vs Predictions Grouped by {group_by_col}', 60)))
        plt.xlabel(group_by_col)
        plt.ylabel('Values')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(path.join(save_dir, f"{prefix}-violin-grouped-by-cancers.png"))
        plt.close()

    def get_fit_aug_args(self, model_type, val_x, val_y):
        kwargs = {}
        if model_type == "keras":
            from tensorflow.keras.callbacks import EarlyStopping
            kwargs["epochs"] = 50
            kwargs["batchsize"] = 32
            kwargs["val_x"] = val_x
            kwargs["val_y"] = val_y
            # Define the EarlyStopping callback
            kwargs["early_stopping"] = EarlyStopping(
                    monitor='val_loss',  # Monitor validation loss
                    patience=5,  # Number of epochs to wait after no improvement
                    restore_best_weights=True  # Restore weights of the best epoch
            )
        return kwargs

    def apply_categorization_to_given_column(self, y_df, given_col, cat_func):
        y_df = pd.DataFrame(y_df)
        y_df[f"{given_col}-category"] = y_df[given_col].apply(cat_func)
        return y_df

    def train_and_save_models_old(self, train_data_dict, val_data_dict, model_name,
                              y_col="IC50", gene_start_col=3, beta=5,
                              use_augmentation=False, if_verbose_K=10,
                              **aug_kwargs):
        """
        Train models for all drugs and save them.
        Args:
            train_data_dict (dict): Dictionary of training data for each drug.
            model_name (str): Name of the regression model.
            gene_start_col (int): Starting column for gene expression features.
            use_augmentation (bool): Whether to apply data augmentation during training.
            **aug_kwargs: Additional arguments for data augmentation.

        1372 Trametinib
        194 Luminespib
        431 Alisertib
        51 Dasatinib
        1373 Dabrafenib
        299 OSI-027
        1392 Bleomycin (10 uM)
        1378 Bleomycin (50 uM)
        190 Bleomycin
        30 Sorafenib
        11 Paclitaxel
        133 Doxorubicin
        """
        all_keys = list(train_data_dict.keys())

        test_keys = all_keys[0: if_verbose_K] + [51, 1373, 299, 1392, 1378, 190, 1372, 194, 431]

        for jj, key in enumerate(test_keys):
            ori_or_aug = "aug" if use_augmentation else "ori"
            [drug, features, y_df] = train_data_dict[key]
            y_df = self.apply_categorization_to_given_column(y_df, "Z score", self.categorize_zscore_3_class)

            self.coll_history = {}
            self._check_col_stats(pd.DataFrame(y_df), ['IC50', 'AUC', 'Z score'],
                    group_by_col='TCGA Classification',
                    prefix=f"{drug.capitalize()}-ori-train",
                    save_dir=self.model_base_dir)

            val_list = val_data_dict[key]
            augmented_datasets = []

            if use_augmentation:
                for beta in [beta]:
                    self.gexmix.beta_param = beta
                    aug_features, aug_y_df = self.augment_data(
                            features.iloc[:, gene_start_col:].reset_index(drop=True),
                            y_df.reset_index(drop=True), if_include_original=True, **aug_kwargs
                    )
                    augmented_datasets.append((beta, aug_features, aug_y_df))
                    self._check_col_stats(
                            pd.DataFrame(aug_y_df), ['IC50', 'AUC', 'Z score'],
                            group_by_col='TCGA Classification',
                            prefix=f"{drug.capitalize()}-Aug-train-beta{beta:.3f}",
                            save_dir=self.model_base_dir)
            else:
                features = features.iloc[:, gene_start_col:].reset_index(drop=True)
                augmented_datasets.append((0, features, y_df))
            # Initialize the model once
            model, model_type = self.get_regression_model_given_name(
                model_name, input_shape=features.shape[1])

            for beta, aug_features, aug_y_df in augmented_datasets:
                y = aug_y_df[y_col]

                self.train_and_save_one_model(model, model_type,
                    key, drug, aug_features, y, model_name, y_col=y_col, beta=beta,
                    val_data_list=val_list, ori_or_aug=ori_or_aug, postfix=f"beta{beta:.3f}",
                    gene_start_col=gene_start_col)
        print("ok")

    def train_and_save_models(self, train_val_data_dict, model_name,
                              y_col="IC50", gene_start_col=3, betas=[5],
                              use_augmentation=False, if_verbose_K=10,
                              **aug_kwargs):
        """
        Train models for all drugs and save them.
        Args:
            train_data_dict (dict): Dictionary of training data for each drug.
            model_name (str): Name of the regression model.
            gene_start_col (int): Starting column for gene expression features.
            use_augmentation (bool): Whether to apply data augmentation during training.
            **aug_kwargs: Additional arguments for data augmentation.

        1372 Trametinib
        194 Luminespib
        431 Alisertib
        51 Dasatinib
        1373 Dabrafenib
        299 OSI-027
        1392 Bleomycin (10 uM)
        1378 Bleomycin (50 uM)
        190 Bleomycin
        30 Sorafenib
        11 Paclitaxel
        133 Doxorubicin
        """
        # Initialize StratifiedKFold
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        need2checktest_keys = [51, 1373, 299, 1392, 1378, 190, 1372, 194, 431]+list(train_val_data_dict.keys())[0:if_verbose_K]
        ##  299, 1392, 1378, 190, 1372, 194, 431 list(train_data_dict.keys())[0:if_verbose_K] +

        # Initialize containers for storing results across drugs
        for jj, drug_key in enumerate(need2checktest_keys):
            # Extract drug data
            drug, train_val_features, train_val_y_df = train_val_data_dict[drug_key]

            ori_or_aug = "aug" if use_augmentation else "ori"
            beta_str = f"{betas[0]:.3f}" if len(betas) == 1 else "multi-beta"
            drug_prefix = f"No{drug_key}-{drug.capitalize()}-{ori_or_aug}-beta{beta_str}"
            print(f"Processing drug {drug_key} {drug.capitalize()} ({jj + 1}/{len(need2checktest_keys)}drugs)")

            train_val_y_df['TCGA Classification'] = train_val_y_df['TCGA Classification'].astype(str)
            if "diagnosis" not in train_val_y_df.columns:
                train_val_y_df['diagnosis'] = train_val_y_df['TCGA Classification'].astype(str)
            else:
                train_val_y_df['diagnosis'] = train_val_y_df['diagnosis'].astype(str)

            self._check_col_stats(
                    pd.DataFrame(train_val_y_df), ['IC50', 'AUC', 'Z score'],
                    group_by_col='TCGA Classification',
                    prefix=f"{drug.capitalize()}-ori-all",
                    save_dir=self.model_base_dir)

            allCV_results = []
            self.coll_history = {}
            # Perform 5-fold cross-validation with leave cell line out data
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_val_y_df["cell_line_name"], train_val_y_df["diagnosis"].values)):
                print(f"{drug}  Fold {fold_idx + 1}")
                fold_drug_results = pd.DataFrame({"Ground Truth": [],
                                                  "Predictions": [],
                                                  "Drug": [],
                                                  "Fold": [],
                                                 "beta": []})

                # Split train and validation data for this fold
                train_features = train_val_features.iloc[train_idx].reset_index(drop=True)
                train_y_df = train_val_y_df.iloc[train_idx].reset_index(drop=True)
                val_features = train_val_features.iloc[val_idx].reset_index(drop=True)
                val_y_df = train_val_y_df.iloc[val_idx].reset_index(drop=True)

                # Initialize model only train one model with all the augmented data
                model, model_type = self.get_regression_model_given_name(
                        model_name, input_shape=train_val_features.shape[1])

                augmented_train_datasets = self.get_training_data_with_augmentation(
                        betas, train_features, train_y_df, drug, gene_start_col,
                        use_augmentation=use_augmentation, **aug_kwargs)

                for beta, train_x, train_y in augmented_train_datasets:
                    # Train and evaluate model
                    fold_ground_truth, fold_predictions = self.train_and_save_one_model(
                            model, model_type, drug_key, drug,
                            train_x, train_y[y_col], model_name,
                            y_col=y_col, beta=beta, val_data_list=[drug, val_features, val_y_df],
                            ori_or_aug=ori_or_aug, postfix=f"fold{fold_idx + 1}-beta{beta:.3f}",
                            gene_start_col=gene_start_col
                    )

                # Store fold results
                fold_drug_results["Ground Truth"] = fold_ground_truth
                fold_drug_results["Predictions"] = fold_predictions
                fold_drug_results["Fold"] = [fold_idx + 1] * len(fold_ground_truth)
                fold_drug_results["Drug"] = [drug] * len(fold_ground_truth)
                fold_drug_results["beta"] = [betas[0]] * len(fold_ground_truth) if len(betas) == 1 else [99] * len(fold_ground_truth)

                fold_drug_results = pd.concat([fold_drug_results, val_y_df], axis=1)

                allCV_results.append(fold_drug_results)

            # Append drug results
            allCV_results_df = pd.concat(allCV_results, ignore_index=True)

            ## concat val meta data to the results_df
            allCV_results_df.to_csv(path.join(self.model_base_dir, f"{drug_prefix}FCV-prediction.csv"), index=True)
            print("Cross-validation results saved to 'cross_validation_results_by_drug.csv'")

    def get_training_data_with_augmentation(self, betas, train_features,
                                            train_y_df, drug, gene_start_col,
                                            use_augmentation=False,
                                            **aug_kwargs):
        aug_by_column = aug_kwargs.get("aug_by_column")
        keys4mix = aug_kwargs.get("keys4mix")

        # Optionally perform data augmentation
        beta_feature_meta_list = []
        if use_augmentation:
            keys4mix += [aug_by_column]
            for beta in betas:
                self.gexmix.beta_param = beta

                if "6_class_zscore" == aug_by_column:
                    train_y_df[aug_by_column] = train_y_df["Z score"].apply(lambda x: categorize_zscore_6_class(x))
                aug_features, aug_y_df = self.augment_data(
                        train_features.iloc[:, gene_start_col:], train_y_df,
                        if_include_original=True, **aug_kwargs
                )
                beta_feature_meta_list.append((beta, aug_features, aug_y_df))
                self._check_col_stats(
                        pd.DataFrame(aug_y_df), ['IC50', 'AUC', 'Z score'],
                        group_by_col="TCGA Classification",
                        prefix=f"{drug.capitalize()}-Aug-train-beta{beta:.3f}",
                        save_dir=self.model_base_dir)
        else:
            train_features = train_features.iloc[:, gene_start_col:]
            beta_feature_meta_list.append((0, train_features, train_y_df))
        return beta_feature_meta_list

    def _check_col_stats(self, y_df, col2check, group_by_col='TCGA Classification',
                         prefix="prefix", save_dir="./"):
        # Create subplots for each metric
        y_df[group_by_col] = y_df[group_by_col].astype(str)
        sample_counts = y_df[group_by_col].astype(str).value_counts()

        # Add counts to TCGA Classification labels
        y_df[f'{group_by_col} (count)'] = y_df[group_by_col].map(
                lambda x: f"{x} (n={sample_counts[x]})"
        )
        # Group by TCGA Classification and calculate mean ground truth
        for col in col2check:
            y_df[f"mean_{col}"] = y_df.groupby(group_by_col)[col].transform('mean')

        # Sort by mean ground truth
        sorted_grouped = y_df.sort_values(f"mean_{col2check[0]}")

        fig, axes = plt.subplots(len(col2check), 1, figsize=(12, 3 * len(col2check)), sharex=True)
        plt.title(f"{prefix}")
        for i, metric in enumerate(col2check):
            sns.boxplot(
                    data=sorted_grouped,
                    x=f'{group_by_col} (count)',
                    y=metric,
                    ax=axes[i]
            )
            axes[i].set_title(col2check[i], fontsize=14)
            axes[i].set_xlabel('' if i < 2 else group_by_col)
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
        fig.tight_layout()
        # Save and close plot
        plt.savefig(path.join(save_dir, f"{prefix}-stats-{group_by_col}.png"))
        plt.close()

    def predict_with_all_models(self, models_dict, gene_expression_sample, y_df, drug="drug", y_col="IC50"):
        """
        Predict drug responses using all trained models for a new gene expression sample.

        Args:
            models (dict): A dictionary of trained models.
            gene_expression_sample (dataframe): a dataframe with rows as samples with simple meta data

        Returns:
            dict: a combined dataframe where columns are predictions from different models, rows are the samples in validation data for one drug.
        """

        # Initialize an empty list to store dataframes
        combined_dataframes = []

        combined_dataframes.append(pd.DataFrame(
                {f"{drug}_GT": y_df[y_col].values},  # Use column values directly
                index=y_df.index  # Ensure consistent indexing
        ))
        for model_drug, model in models_dict.items():
            drug_predicts = model.predict(gene_expression_sample.values)
            if drug in model_drug:
                predictions_df = pd.DataFrame(
                    drug_predicts, index=gene_expression_sample.index, columns=[f"{drug}_pred"])
            else:
                predictions_df = pd.DataFrame(
                        drug_predicts, index=gene_expression_sample.index,
                        columns=[f"{model_drug}_pred"])
            # Append the modified DataFrame to the list
            combined_dataframes.append(predictions_df)

        # Concatenate all DataFrames along columns (axis=1)
        combined_dataframe = pd.concat(combined_dataframes, axis=1)

        return combined_dataframe

    def load_saved_models(self, name_str_pattern, model_saved_dir):
        """
        Load all saved models into memory.

        Returns:
            dict: A dictionary with drug names as keys and models as values.
        """
        models = {}
        model_names = glob(f"{model_saved_dir}/{name_str_pattern}")

        for model_file in model_names:
            drug_no = model_file.split("No")[-1].split("_")[0]
            drug_name = model_file.split("drug")[-1].split(".pkl")[0]
            models[f"{drug_no}_{drug_name}"] = joblib.load(model_file)
        return models

    def get_regression_model_given_name(self, model_name, input_shape=400, gamma=2.0, alpha=0.25,
                                        droprate=0.58, class_weight={0: 0.5, 1: 0.5}, **kwargs):
        from sklearn.linear_model import ElasticNet, ElasticNetCV, MultiTaskElasticNetCV, Lasso, \
            LinearRegression, BayesianRidge, TweedieRegressor
        if model_name.lower() == "bayesianridge":
            model = BayesianRidge()
            model_type = "sklearn"
        elif model_name.lower() == "xgboost":
            model = xgb.XGBRegressor(
                    objective='reg:squarederror',  # Regression task
                    learning_rate=0.1,  # Step size
                    max_depth=6,  # Maximum tree depth
                    n_estimators=100,  # Number of boosting rounds
                    subsample=0.8,  # Fraction of samples per tree
                    colsample_bytree=0.8,  # Fraction of features per tree
                    random_state=42  # For reproducibility
            )
            model_type = "sklearn"
        elif model_name.lower() == "lasso":
            model = Lasso(
                    alpha=1.0,
                    fit_intercept=True,
                    precompute=False,
                    copy_X=True,
                    max_iter=10000,
                    tol=1e-4,
                    warm_start=False,
                    positive=False,
                    random_state=899,
                    selection="cyclic")
            model_type = "sklearn"
        elif model_name.lower() == "linearregression":
            model = LinearRegression()
            model_type = "sklearn"
        elif model_name.lower() == "tweedieregressor":
            model = TweedieRegressor(power=1, alpha=0.5, link='log')
            model_type = "sklearn"
        elif model_name.lower() == "fnn":
            model = Sequential(
                    [
                            Dense(512, input_dim=input_shape, activation='relu'),
                            BatchNormalization(),
                            Dropout(0.8),
                            Dense(64, activation='relu'),
                            BatchNormalization(),
                            Dropout(0.8),
                            Dense(1)
                            # Use sigmoid for multi-label classification
                    ])
            # Compile the model
            model.compile(
                    optimizer=Adam(learning_rate=0.0005), loss='mse',
                    metrics=['mse'])
            model_type = "keras"
        return model, model_type

    def get_classifier_model_given_name(self, model_name, input_shape=400, output_shape=3, gamma=2.0,
                                        alpha=0.25,
                                        droprate=0.58, class_weight={0: 0.5, 1: 0.5}, **kwargs):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import ElasticNet, ElasticNetCV, MultiTaskElasticNetCV, Lasso, \
            LinearRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.neighbors import KNeighborsClassifier

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
                    hidden_layer_sizes=32, activation="relu", validation_fraction=0.2,
                    max_iter=10000)
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

    def evaluate_performance(self, val_data, predictions, train_data):
        """
        Evaluate prediction performance on validation data, considering overlap of cell lines.

        Args:
            val_data (DataFrame): Validation data with columns ['cell_line', 'drug', 'true_response'].
            predictions (DataFrame): Predictions with columns ['cell_line', 'drug', 'predicted_response'].
            train_data (DataFrame): Training data with columns ['cell_line', 'drug', 'response'].

        Returns:
            dict: A dictionary containing performance metrics for overlapping and non-overlapping cell lines.
        """
        # Merge validation data with predictions
        val_data = val_data.merge(predictions, on=['cell_line', 'drug'])

        # Identify overlapping and non-overlapping cell lines
        train_cell_lines = set(train_data['cell_line'])
        val_data['overlap'] = val_data['cell_line'].isin(train_cell_lines)

        # Partition validation data
        overlapping = val_data[val_data['overlap']]
        non_overlapping = val_data[~val_data['overlap']]

        # Helper function to compute metrics
        def compute_metrics(true_values, pred_values):
            mse = mean_squared_error(true_values, pred_values)
            mae = mean_absolute_error(true_values, pred_values)
            pearson_corr, _ = pearsonr(true_values, pred_values)
            spearman_corr, _ = spearmanr(true_values, pred_values)
            return {
                    'MSE': mse,
                    'MAE': mae,
                    'Pearson Correlation': pearson_corr,
                    'Spearman Correlation': spearman_corr
            }

        # Compute metrics for overlapping and non-overlapping cell lines
        metrics_overlapping = compute_metrics(
                overlapping['true_response'], overlapping['predicted_response'])
        metrics_non_overlapping = compute_metrics(
                non_overlapping['true_response'], non_overlapping['predicted_response'])

        # Combine results
        results = {
                'Overlapping Cell Lines': metrics_overlapping,
                'Non-Overlapping Cell Lines': metrics_non_overlapping
        }

        return results
