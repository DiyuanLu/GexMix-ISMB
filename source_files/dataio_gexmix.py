from __future__ import division
from __future__ import print_function
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from os import path, makedirs, listdir
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from scipy.stats import zscore

import plotting_pan as Plot


# from drug_loader import UDF_drug

def load_data_into_dataset(args):
	"""
	Load from the mat file
	tag sampleID
	create tf datasets
	:param args:
	:return:
	"""
	
	if args.num_genes == 1024:
		with open("../data/1_filter_CCLE_with_TCGA/full_data_labels_gene_1024.pickle", "rb") as file:
			related_labels_dict = pickle.load(file)
			normalized_data = related_labels_dict["rnaseq"]
			del related_labels_dict["rnaseq"]
			args.num_genes = normalized_data.shape[1]
			args.data_shape = [args.num_genes, 1, 1]
	else:
		related_labels_dict, normalized_data = load_filtered_TCGA_and_CCLE_dataset(args)
	# after this, all labesl should be in one-hot encoding
	# https://github.com/mstrazar/iONMF
	features_train_val_dict, labels_train_val_dict = tag_and_train_val_split(args, related_labels_dict,
	                                                                         normalized_data)
	
	# # apply mixup to training set if needed
	related_labels_aug, normal_data_aug = apply_mixup(features_train_val_dict["train"],
	                                                  labels_train_val_dict["train"],
	                                                  num_valid_class=args.valid_classes,
	                                                  label2name=args.label2name,
	                                                  if_include_original=args.if_include_original,
	                                                  num2aug=args.num2aug, num2mix=args.num2mix,
	                                                  cancer_colors=args.cancer_colors,
	                                                  if_use_remix=args.if_use_remix,
	                                                  if_stratify_mix=args.if_stratify_mix,
	                                                  num_total_classes=args.num_classes,
	                                                  save_dir=args.save_dir)

	related_labels_aug["tcga_hot_labels"] = np.append(related_labels_aug["tcga_hot_labels"], np.zeros((
			related_labels_aug["tcga_hot_labels"].shape[0], args.num_classes - related_labels_aug[
				"tcga_hot_labels"].shape[1])), axis=1)
	features_train_val_dict["train"] = normal_data_aug
	labels_train_val_dict["train"] = related_labels_aug
	
	for name in ["train", "val", "test"]:
		labels_train_val_dict[name]["features"] = features_train_val_dict[name]
	
	args.dataset_tensor_names = {"train": ["features"], "val": ["features"], "test": ["features"]}
	data_sets_dict = {}
	for name in ["train", "val"]:
		ds_element_list = [features_train_val_dict[name]]
		for key in args.input_tensor_names:
			if key in labels_train_val_dict[name].keys():
				ds_element_list.append(labels_train_val_dict[name][key])
				args.dataset_tensor_names[name].append(key)
		data_sets_dict[name] = create_dataset(tuple(ds_element_list), batch_size=args.batch_size,
		                                      train_or_test=name)
	
	for name in ["test"]:
		ds_element_list = [features_train_val_dict[name]]
		for key in labels_train_val_dict[name].keys():
			if key != "features":
				ds_element_list.append(labels_train_val_dict[name][key])
				args.dataset_tensor_names[name].append(key)
		data_sets_dict[name] = create_dataset(tuple(ds_element_list), batch_size=args.batch_size,
		                                      train_or_test=name)
	try:  # with partial data, there might not be enough source to plot
		# visualize the original data
		reencoded_tcga_labels = np.array([ele for ele in list(labels_train_val_dict["diagnosis"][
			                                                      "train"])])
		projection = Plot.visualize_aligned_marked_embedding_with_reduced_dimension(np.squeeze(features_train_val_dict[
			                                                                                       "train"]),
		                                                                            {"source_labels":
			                                                                             labels_train_val_dict[
				                                                                             "source_labels"]["train"],
		                                                                             "diagnosis": reencoded_tcga_labels},
		                                                                            args,
		                                                                            title="",
		                                                                            cancer_colors=args.cancer_colors,
		                                                                            prefix=f"Original training data "
		                                                                                   f"colored by cancer types",
		                                                                            figsize=[8, 6],
		                                                                            vis_mode="umap", size=15,
		                                                                            save_dir=path.join(
				                                                                            args.save_dir,
				                                                                            "Latent umaps"))
	except:
		print("Original data can't be plotted")
	return data_sets_dict["train"], data_sets_dict["val"], data_sets_dict["test"], args


def load_data_from_csv(args):
	if args.num_genes == 1024:
		with open("../data/1_filter_CCLE_with_TCGA/full_data_labels_gene_1024.pickle", "rb") as file:
			related_labels_dict = pickle.load(file)
			normalized_data = related_labels_dict["rnaseq"]
			del related_labels_dict["rnaseq"]
			args.num_genes = normalized_data.shape[1]
			args.data_shape = [args.num_genes, 1, 1]
	else:
		related_labels_dict, normalized_data = load_filtered_TCGA_and_CCLE_dataset(args)
	# after this, all labesl should be in one-hot encoding
	# https://github.com/mstrazar/iONMF
	features_train_val_dict, labels_train_val_dict = tag_and_train_val_split(args, related_labels_dict,
	                                                                         normalized_data)
	
	return features_train_val_dict, labels_train_val_dict


def train_val_split_ignore_singletons(X, y, test_size=0.2, random_state=None):
	"""
	Here X is
	:param X: sample_ids
	:param y: labels: array
	:param test_size:
	:param random_state:
	:return:
	"""
	# Get unique class labels and their counts
	unique_classes, class_counts = np.unique(y, return_counts=True)
	
	# Filter out classes with only one sample
	valid_classes = unique_classes[class_counts > 1]
	mask = np.isin(y, valid_classes)
	
	# Perform stratified train-test split
	X_train, X_val, y_train, y_val = train_test_split(
			X[mask], y[mask],
			test_size=test_size,
			random_state=random_state,
			stratify=y[mask]
	)
	
	# Assert that there are no repeated indices in train and val
	assert len(np.intersect1d(X_train, X_val)) == 0, "Error: Repeated indices between train and val sets."
	
	return X_train, X_val, y_train, y_val


def  tag_and_train_val_split(args, labels_dict, normal_data, indices_file=None):
	"""
	assign a sample iD to samples for future investigation
	:param args:
	:param labels_dict: "source_labels", "site_labels", "tcga_hot_labels"
	:param normal_data:
	:param indices_file: a pickle file with saved sample indices for train, val, test-dataset
	:return:
	"""
	sample_inds = np.arange(len(normal_data))
	labels_dict["sample_ids"] = sample_inds
	
	if not indices_file:  # this is training, no saved files with indices for train, val, test-dataset
		train_val_inds, test_inds, _, _ = train_val_split_ignore_singletons(sample_inds,
		                                                                    np.argmax(labels_dict["tcga_hot_labels"],
		                                                                              axis=1),
		                                                                    test_size=args.test_ratio)  # ,random_state=88
		train_inds = train_val_inds
		val_inds = test_inds
	
	else:  # there is a saved file with all indices
		# save the summary of the train, val, and test data
		file = open(indices_file, 'rb')
		saved_indices = pickle.load(file)
		
		# train_inds = saved_indices["sample_ids"]["train"]
		val_inds = saved_indices["val"]["sample_ids"]
		test_inds = saved_indices["test"]["sample_ids"]
		train_inds = labels_dict["sample_ids"]
	
	features_train_val_dict = {}
	inds_train_val_dict = {}
	labels_train_val_dict = {"train": {key: [] for key in labels_dict.keys()},
	                         "val": {key: [] for key in labels_dict.keys()},
	                         "test": {key: [] for key in labels_dict.keys()}}
	
	for key, indices in zip(["train", "val", "test"], [train_inds, val_inds,  test_inds]):
		overall_inds = fill_last_batch_get_inds(args.batch_size, indices)
		inds_train_val_dict[key] = overall_inds
		features_train_val_dict[key] = normal_data[overall_inds]

	# source label and sample id
	for key in labels_dict.keys():
		if key == "source_labels":
			labels_train_val_dict["train"][key] = np.eye(args.num_sources)[labels_dict[key][inds_train_val_dict[
				"train"]]]
			labels_train_val_dict["val"][key] = np.eye(args.num_sources)[labels_dict[key][inds_train_val_dict["val"]]]
			labels_train_val_dict["test"][key] = np.eye(args.num_sources)[labels_dict[key][inds_train_val_dict["test"]]]
		elif key == "tumor_percent":
			labels_train_val_dict["train"][key] = labels_dict[key][inds_train_val_dict[
				"train"]] / 100.
			labels_train_val_dict["val"][key] = labels_dict[key][inds_train_val_dict["val"]] / 100.
			labels_train_val_dict["test"][key] = labels_dict[key][inds_train_val_dict["test"]] / 100.
		else:
			labels_train_val_dict["train"][key] = labels_dict[key][inds_train_val_dict[
				"train"]]
			labels_train_val_dict["val"][key] = labels_dict[key][inds_train_val_dict["val"]]
			labels_train_val_dict["test"][key] = labels_dict[key][inds_train_val_dict["test"]]
	
	return features_train_val_dict, labels_train_val_dict


def fill_last_batch_get_inds(batch_size, original_data):
	need2add_num = batch_size - len(original_data) % batch_size
	need2add_inds = np.random.choice(original_data, need2add_num)
	total_filled_last_batch = list(original_data) + list(need2add_inds)
	return total_filled_last_batch

#
# def load_only_filtered_CCLE_dataset(args):
# 	"""
# 	Load the filtered out data with the shared gene-list between TCGA and CCLE
# 	load the updated CCLE_sample_info_with_tcga_label_imputation, generated with
# 	here the ccle_sample_info_file is generated by impute_tcga_labels_of_CCLE_with_annotation_file(
# 	original_tcga_sample_info, original_ccle_sample_info)
# 	:param args:
# 	:return:
# 	"""
#
# 	ccle_data, ccle_sample_info = load_data_and_info_given_filenames(args.ccle_file, args.ccle_sample_info_file,
# 	                                                                 if_to_cluster=args.if_to_cluster,
# 	                                                                 fullset_train=args.fullset_train,
# 	                                                                 dataset_name="ccle",
# 	                                                                 data_root=args.data_dir)
#
# 	filename = "../data/1_filter_CCLE_with_TCGA/header_with_used_1024-gene_list.csv"
# 	all_columns = pd.read_csv(filename)
# 	assert num2select == all_columns.shape[1], "number of features doens't match"
# 	selected_columns = all_columns[0:num2select]
# 	comb_data = pd.concat([tcga_data, ccle_data], axis=0)
# 	stddata = np.std(comb_data, axis=0)
# 	meandata = np.mean(comb_data, axis=0)
#
# 	selcted_genes_w_mean = pd.DataFrame(columns=["mean", "std", "index"])
# 	selcted_genes_w_mean["index"] = selected_columns
# 	selcted_genes_w_mean["mean"] = meandata[selected_columns]
# 	selcted_genes_w_mean["std"] = stddata[selected_columns]
# 	selcted_genes_w_mean.to_csv(
# 			path.join(args.data_dir, f"selected_top{len(selected_columns)}_based_on_{select_mode}.csv"), index=False)
#
# 	selected_comb_data = comb_data[selected_columns]
#
# 	selected_comb_data.iloc[0:5].to_csv(path.join(args.save_dir, f"header_with_used"
# 	                                                             f"_{selected_comb_data.shape[1]}-gene_list.csv"))
# 	args.num_genes = selected_comb_data.shape[1]
# 	args.data_shape = [args.num_genes, 1, 1]
# 	# zscore normalzation
# 	normal_data = zscore(selected_comb_data.values, axis=1)
#
# 	related_labels = get_joint_labels_from_datasets([ccle_sample_info],
# 	                                                num_classes=args.num_classes,
# 	                                                tcga_name2int=args.tcga_name2int, source_lb_assign=[1])
#
# 	# with open(path.join(args.save_dir, f'partial_data_labels_{normal_data.shape[0]}_{normal_data.shape[1]}.pickle'), 'wb') as handle:
# 	#     related_labels["rnaseq"] = normal_data
# 	#     pickle.dump(related_labels, handle)
# 	return related_labels, normal_data


def load_filtered_TCGA_and_CCLE_dataset(args):
	"""
	Load the filtered out data with the shared gene-list between TCGA and CCLE
	load the updated CCLE_sample_info_with_tcga_label_imputation, generated with
	here the ccle_sample_info_file is generated by impute_tcga_labels_of_CCLE_with_annotation_file(
	original_tcga_sample_info, original_ccle_sample_info)
	:param args:
	:return:
	"""
	
	ccle_data, ccle_sample_info = load_data_and_info_given_filenames(args.ccle_file, args.ccle_sample_info_file,
	                                                                 if_to_cluster=args.if_to_cluster,
	                                                                 fullset_train=args.fullset_train,
	                                                                 dataset_name="ccle",
	                                                                 data_root=args.data_dir)
	
	tcga_data, tcga_sample_info = load_data_and_info_given_filenames(args.tcga_file, args.tcga_sample_info_file,
	                                                                 if_to_cluster=args.if_to_cluster,
	                                                                 fullset_train=args.fullset_train,
	                                                                 dataset_name="tcga",
	                                                                 data_root=args.data_dir)
	
	selected_comb_data, selected_columns, args = select_subset_features(args,
	                                                                    {"tcga": tcga_data, "ccle": ccle_data},
	                                                                    num2select=args.num2select,
	                                                                    select_mode=args.select_mode)
	
	args.num_genes = selected_comb_data.shape[1]
	args.data_shape = [args.num_genes, 1, 1]
	# zscore normalzation
	normal_data = zscore(selected_comb_data.values, axis=1)
	
	related_labels = get_joint_labels_from_datasets([tcga_sample_info, ccle_sample_info], num_classes=args.num_classes,
	                                                tcga_name2int=args.tcga_name2int, source_lb_assign=[0, 1])
	
	# with open(path.join(args.save_dir, f'partial_data_labels_{normal_data.shape[0]}_{normal_data.shape[1]}.pickle'), 'wb') as handle:
	#     related_labels["rnaseq"] = normal_data
	#     pickle.dump(related_labels, handle)
	return related_labels, normal_data


def get_union_of_selected_features_from_both_datasets(args, ccle_data, tcga_data):
	mean_tcga = np.mean(tcga_data, axis=0)
	mean_ccle = np.mean(ccle_data, axis=0)
	std_tcga = np.std(tcga_data, axis=0)
	std_ccle = np.std(ccle_data, axis=0)
	uniq_genes, counts = np.unique(list(tcga_data.columns) + list(ccle_data.columns),
	                               return_counts=True)
	uniq_gene_df = pd.DataFrame(np.zeros((1, len(uniq_genes))), columns=uniq_genes)
	gene_in_both = uniq_genes[counts == 2]
	gene_in_one = uniq_genes[counts == 1]
	uniq_gene_df[gene_in_both] = (mean_tcga[gene_in_both] + mean_ccle[gene_in_both]) / 2
	for col in gene_in_one:
		if col in tcga_data.columns:
			uniq_gene_df[col] = mean_tcga[col]
		elif col in ccle_data.columns:
			uniq_gene_df[col] = mean_ccle[col]
	sorted_mean_df = uniq_gene_df[uniq_gene_df.iloc[0].sort_values(ascending=False).index]
	
	return sorted_mean_df


def select_subset_features(args, comb_data_dict, num2select=None, select_mode="std"):
	"""

	:param args:
	:param comb_data_dict:
	:param num2select:
	:param base:
	:return:
	"""
	if "tcga" in comb_data_dict.keys():
		tcga_data = comb_data_dict["tcga"]
	if "ccle" in comb_data_dict.keys():
		ccle_data = comb_data_dict["ccle"]
	
	if num2select is None:
		selected_comb_data = pd.concat([tcga_data, ccle_data], axis=0)
	else:
		if select_mode == "std":
			mean_tcga = np.mean(tcga_data, axis=0)
			mean_ccle = np.mean(ccle_data, axis=0)
			std_tcga = np.std(tcga_data, axis=0)
			std_ccle = np.std(ccle_data, axis=0)
			
			# Get the top 50 columns with the highest standard deviation from each dataset
			top_std_tcga_cols = std_tcga.nlargest(args.num2select).index
			top_std_ccle_cols = std_ccle.nlargest(args.num2select).index
			
			# Take the union of the selected columns
			selected_columns = list(set(top_std_tcga_cols) | set(top_std_ccle_cols))
			
			joint_top_set = std_tcga[selected_columns] + std_ccle[selected_columns]
			final_selected_columns = joint_top_set.nlargest(args.num2select).index
			
			# Combine the mean values from both datasets and sort them in descending order
			combined_mean = mean_tcga[final_selected_columns] + mean_ccle[final_selected_columns]
			combined_std = std_tcga[final_selected_columns] + std_ccle[final_selected_columns]
			ranked_columns = combined_mean.sort_values(ascending=False).index
			
			selcted_genes_w_mean = pd.DataFrame(columns=["mean", "std", "index"])
			selcted_genes_w_mean["index"] = ranked_columns
			selcted_genes_w_mean["mean"] = combined_mean[ranked_columns].values
			selcted_genes_w_mean["std"] = combined_std[ranked_columns].values
			selcted_genes_w_mean.to_csv(
					path.join(args.save_dir, f"selected_top{len(final_selected_columns)}_based_on_{select_mode}.csv"),
					index=False)
			
			selected_comb_data = pd.concat([tcga_data[ranked_columns], ccle_data[ranked_columns]], axis=0)
		elif select_mode == "from_file":
			filename = "../data/1_filter_CCLE_with_TCGA/header_with_used_1024-gene_list.csv"
			# filename = "../data/relevant_with_Daksh/ordered_gene_list_all_nor.pkl"
			# with open(filename, 'rb') as f:
			#     all_columns = pickle.load(f)
			all_columns = pd.read_csv(filename)
			selected_columns = all_columns[0:num2select]
			comb_data = pd.concat([tcga_data, ccle_data], axis=0)
			stddata = np.std(comb_data, axis=0)
			meandata = np.mean(comb_data, axis=0)
			
			selcted_genes_w_mean = pd.DataFrame(columns=["mean", "std", "index"])
			selcted_genes_w_mean["index"] = selected_columns
			selcted_genes_w_mean["mean"] = meandata[selected_columns]
			selcted_genes_w_mean["std"] = stddata[selected_columns]
			selcted_genes_w_mean.to_csv(
				path.join(args.data_dir, f"selected_top{len(selected_columns)}_based_on_{select_mode}.csv"),
				index=False)
			
			selected_comb_data = comb_data[selected_columns]
	
	selected_comb_data.iloc[0:5].to_csv(path.join(args.save_dir, f"header_with_used"
	                                                             f"_{selected_comb_data.shape[1]}-gene_list.csv"))
	# selected_comb_data.iloc[0:5].to_csv(path.join("../data/1_filter_CCLE_with_TCGA, f"header_with_used"
	#                                                              f"_{selected_comb_data.shape[1]}-gene_list.csv"))
	return selected_comb_data, selected_columns, args


def get_joint_labels_from_datasets(meta_data_list, num_classes=45, tcga_name2int={"ACC": 0}, source_lb_assign=[0, 1]):
	"""
	"""
	source_labels = []
	site_labels = []
	disesae_labels = []
	disesae_labels_imputed = []
	tumor_percent = []
	normal_percent = []
	sample_id = []
	# TODO: how to make the columns to extract dynamic and extenable? because which elements are selected affects
	#  later in the dataset initialization to handle each element individually. Need a function that can take
	#  arbitory number and columns from the sample_meta_info
	for source_lb, data_meta in zip(source_lb_assign, meta_data_list):
		source_labels += [source_lb] * len(data_meta)
		site_labels += list(data_meta["primary_site"])
		disesae_labels += list(data_meta["diagnosis_b4_impute"])
		disesae_labels_imputed += list(data_meta["diagnosis"])
		tumor_percent += list(data_meta["tumor_percent"])
		normal_percent += list(data_meta["normal_percent"])
		sample_id += list(data_meta["sample_id"])
	
	uniq_diagnosis, uniq_inds, counts = np.unique(disesae_labels, return_index=True, return_counts=True)
	for diag in uniq_diagnosis:
		diag_inds = np.where(np.array(disesae_labels) == diag)[0]
		sources = np.unique(np.array(source_labels)[diag_inds])
		print(f"{diag} has {len(diag_inds)} from {np.array(['TCGA', 'CCLE'])[sources]}")
	related_labels = {"source_labels": np.array(source_labels),
	                  "site_labels": np.array(site_labels),
	                  "diagnosis_labels_b4_impute": np.array(disesae_labels),
	                  "diagnosis": np.array(disesae_labels_imputed),
	                  "tumor_percent": np.array(tumor_percent) / 100.,
	                  "normal_percent": np.array(normal_percent),
	                  "sample_id": np.array(sample_id),
	                  }
	
	#  get tcga int labels. First replace COAD/READ labels
	related_labels["diagnosis_labels_b4_impute"] = np.array([ele.split("-")[-1] for ele in
	                                                         related_labels["diagnosis_labels_b4_impute"]])
	for col in ["diagnosis_labels_b4_impute", "diagnosis"]:
		temp_inds = np.where(related_labels[col] == "COAD/READ")[0]
		related_labels[col][temp_inds] = np.array(["COAD", "READ"])[np.random.randint(0, 2, len(temp_inds))]
	
	tcga_int_labels = np.array(
			[tcga_name2int[name] for name in related_labels["diagnosis"]])
	related_labels["tcga_hot_labels"] = np.eye(num_classes)[tcga_int_labels]  # here num_classes=47
	return related_labels


def load_data_and_info_given_filenames(data_filename, sample_info_filename, if_to_cluster=False,
                                       fullset_train=True, dataset_name="tcga", data_root="../data"):
	"""
	# TCGA: [10374, Name+17713],
	# CCLE [1248, Name+17713], originally 1249, discarded one
	:param data_filename:
	:param sample_info_filename:
	:param if_to_cluster:
	:param data_root:
	:return:
	"""
	if not if_to_cluster:
		if fullset_train:
			data_sample_info = pd.read_csv(path.join(data_root, sample_info_filename))
			loaded_data = pd.read_csv(path.join(data_root, data_filename))  # [10374, Name+17713]
		else:
			num2select = 2000
			data_sample_info = pd.read_csv(path.join(data_root, sample_info_filename), nrows=num2select)
			loaded_data = pd.read_csv(path.join(data_root, data_filename), nrows=num2select)  # [10374, Name+17713]
	else:
		data_sample_info = pd.read_csv(path.join(data_root, sample_info_filename))
		loaded_data = pd.read_csv(path.join(data_root, data_filename))  # [10374, Name+17713]
	
	assert len(loaded_data) == len(data_sample_info), "Numbers of TCGA sample don't match!"
	
	site_col_name = [ele for ele in ["gdc_cases.project.primary_site", "added_transformed_lineage"] if ele in
	                 data_sample_info.columns][0]
	disesae_col_name = [ele for ele in ["gdc_cases.project.project_id", "added_tcga_labels_b4_impute"] if ele in
	                    data_sample_info.columns][0]  # 'TCGA-ESCA
	disesae_labels_imputed_col_name = [ele for ele in ["diagnosis", "added_tcga_labels"] if ele in
	                                   data_sample_info.columns][0]  # ESCA
	
	# [print(col, np.unique(data_sample_info[col], return_counts=True)) for col in data_sample_info.columns]
	
	if dataset_name == "ccle":  # unify the column name of genes between T
		loaded_data.columns = [ele.split(" (")[0] for ele in loaded_data.columns]
		data_sample_info.insert(data_sample_info.shape[1], "tumor_percent", 100 * np.ones(len(data_sample_info)))
		data_sample_info.insert(data_sample_info.shape[1], "normal_percent", np.zeros(len(data_sample_info)))
		# add the following two cols to match the format of TCGA dataset
		tumor_percent = "tumor_percent"
		normal_percent = "normal_percent"
		sample_id = "DepMap_ID"
		
		# correct one misclassification
		mis_clf_ind = np.where(data_sample_info["stripped_cell_line_name"] == "HCC1588")[0]
		data_sample_info.at[mis_clf_ind[0], "added_tcga_labels_b4_impute"] = "NSCLC"
		data_sample_info.at[mis_clf_ind[0], "added_tcga_labels"] = "NSCLC"
	# Another misclassification
	# mis_clf_ind = np.where(data_sample_info["stripped_cell_line_name"] == "SKNEP1")[0]
	# data_sample_info.iloc[mis_clf_ind[0], "diagnosis"] = "NSCLC"
	# data_sample_info.iloc[mis_clf_ind[0], "diagnosis_b4_impute"] = "NSCLC"
	
	if dataset_name == "tcga":
		tumor_percent = "cgc_slide_percent_tumor_nuclei"
		normal_percent = "cgc_slide_percent_normal_cells"
		sample_id = "sample_id"
	
	new_column_names = {
			site_col_name: 'primary_site',
			disesae_col_name: 'diagnosis_b4_impute',
			disesae_labels_imputed_col_name: "diagnosis",
			tumor_percent: 'tumor_percent',  # newly added for downstream task
			normal_percent: 'normal_percent',
			sample_id: 'sample_id',
	}
	data_sample_info.rename(columns=new_column_names, inplace=True)
	data_sample_info['diagnosis_b4_impute'] = [ele.split("-")[-1] for ele in data_sample_info['diagnosis_b4_impute']]
	if "Unnamed: 0" in loaded_data.columns:
		loaded_data.drop(["Unnamed: 0"], axis=1, inplace=True)
	if "Unnamed: 0" in data_sample_info.columns:
		data_sample_info.drop(["Unnamed: 0"], axis=1, inplace=True)
	
	return loaded_data, data_sample_info


def impute_tcga_labels_of_CCLE_with_annotation_file(tcga_sample_info, ccle_sample_info):
	"""
	One time use function.
	1. used once to impute TCGA labels of CCLE data with the help of "Cell_lines_annotations_20181226.txt" file.
	Out of 1756 cell ines in CCLE, the annotation file has 1461 shared cell line IDs with tcga_labels. The rest we
	imputated from the existing annotation.
	2. process the site info such that the CCLE and TCGA datasets have a some shared names, mainly dealing with
	letter cases and different names such as gastric-stomach
	:param tcga_sample_info:
	:param ccle_sample_info:
	:return:
	"""
	cell_line_annotation = pd.read_table("../data/cell_line_with_tumor/from_alex_Cell_lines_annotations_20181226.txt")
	cell_line_anno_tcga_code = cell_line_annotation["tcga_code"]
	cell_line_anno_CCLE_ID = cell_line_annotation["CCLE_ID"]  # 1461 annotated cell line
	
	tcga_sites = tcga_sample_info["gdc_cases.project.primary_site"]
	tcga_disease = tcga_sample_info["gdc_cases.project.name"]
	
	ccle_sites = ccle_sample_info["lineage"]  # tumor site info
	ccle_subtypes = ccle_sample_info["lineage_subtype"]  # "lineage_subtype", close to TCGA labels
	ccle_names = ccle_sample_info["CCLE_Name"]  # "CCLE_Name"
	
	# 1.  get the tcga_labels from those shared CCLE_ID cell lines from the annotation file
	# new_frame = pd.DataFrame(columns=["CCLE_Name", "tcga_labels", "original_lineage_subtype", "manual_impute"])
	new_frame = ccle_sample_info.copy()
	new_frame.insert(3, 'added_tcga_labels_b4_impute', np.zeros(len(new_frame)))
	for ii, id in enumerate(ccle_names.values):
		if id in list(cell_line_anno_CCLE_ID.values):  # if the ccle_ID is also in anno file, then assign
			current_ind = np.where(id == cell_line_anno_CCLE_ID.values)[0]
			new_frame.at[ii, "added_tcga_labels_b4_impute"] = cell_line_anno_tcga_code.values[current_ind[0]]
	new_frame = new_frame.fillna(value=0)
	new_frame.to_csv(path.join("../data/1_filter_CCLE_with_TCGA",
	                           "CCLE_sample_info_with_TCGA_labels_from_alex_annotation_b4_imputation_step1_1248_new"
	                           ".csv"),
	                 index=False)  # 1248 samples
	# 2. we manually imputed tcga_labels for some obvious cell lines. then we add the manual_impute indicator
	# save the ccle sample info with manual_impute col
	new_data_with_imputation = pd.read_csv(path.join("../data/1_filter_CCLE_with_TCGA",
	                                                 "CCLE_expression_TCGA_labels_from_alex_anno_with_imputation_col"
	                                                 "_no_delete.csv"))  # [1756, 5]
	
	ccle_sample_info_with_tcga_label_imputation = new_frame.copy()
	ccle_sample_info_with_tcga_label_imputation.insert(4, 'added_tcga_labels',
	                                                   np.ones(len(new_frame)) * 44)
	ccle_sample_info_with_tcga_label_imputation.insert(5, 'added_if_manual_impute', np.ones(len(new_frame)) * 55)
	
	for jj in range(len(ccle_sample_info_with_tcga_label_imputation)):
		ind_in_new_data_with_imputation = list(new_data_with_imputation["CCLE_Name"].values).index(new_frame[
			                                                                                           "CCLE_Name"][jj])
		matched_cell_line_in_imputation_file = new_data_with_imputation.iloc[ind_in_new_data_with_imputation, :]
		matched_cell_line_in_imputation_file.fillna(0)
		assert new_frame["CCLE_Name"][jj] == matched_cell_line_in_imputation_file["CCLE_Name"], "CCLE_name is not " \
		                                                                                        "matching!"
		ccle_sample_info_with_tcga_label_imputation.at[jj, "added_tcga_labels"] = \
			matched_cell_line_in_imputation_file["tcga_labels"]
		
		if matched_cell_line_in_imputation_file["tcga_labels"] != matched_cell_line_in_imputation_file[
			"tcga_label_b4_impute"]:
			ccle_sample_info_with_tcga_label_imputation["added_if_manual_impute"][jj] = True
		else:
			ccle_sample_info_with_tcga_label_imputation["added_if_manual_impute"][jj] = False
	
	# here is where the mistake was made. The really ccle_sample_info only have 1248 valid strains,
	# but new_data_with_imputation has 1756 strains. NOT MATCHING!!!
	ccle_sample_info_with_tcga_label_imputation.to_csv(path.join("../data/1_filter_CCLE_with_TCGA",
	                                                             "CCLE_sample_info_with_imputed_TCGA_labels_from_alex_annotation_step2_1248"
	                                                             ".csv"),
	                                                   index=False)
	
	# 3. get shared set of the primary site for both TCGA and CCLE
	transformed_tcga_sites = [ele.lower() for ele in tcga_sites.values]
	tcga_sample_info["gdc_cases.project.primary_site_lower_case"] = transformed_tcga_sites
	transformed_ccle_sites = [" ".join(ele.lower().split("_")[0:]) for ele in ccle_sites]
	
	# combine the original CCLE sample info with the tcga_label imputation
	ccle_sample_info_with_tcga_label_imputation["added_transformed_lineage"] = [""] * len(
			ccle_sample_info_with_tcga_label_imputation)
	new_transfored_ccle_sites_with_ref_to_tcga = []
	for site in transformed_ccle_sites:
		if site == "urinary tract":
			site = "bladder"
		elif site == "central nervous system":
			site = "brain"
		elif site == "adrenal cortex":
			site = "adrenal gland"
		elif site == "gastric":
			site = "stomach"
		elif site == "upper aerodigestive":
			site = "head and neck"
		elif site == "lymphocyte":
			site = "lymph nodes"
		new_transfored_ccle_sites_with_ref_to_tcga.append(site)
	
	ccle_sample_info_with_tcga_label_imputation[
		"added_transformed_lineage"] = new_transfored_ccle_sites_with_ref_to_tcga
	uniq_ccle_new_sites = np.unique(new_transfored_ccle_sites_with_ref_to_tcga)
	uniq_tcga_new_sites = np.unique(transformed_tcga_sites)
	
	uniq_sites, indexs, counts = np.unique(list(uniq_ccle_new_sites) + list(
			uniq_tcga_new_sites), return_index=True, return_counts=True)
	
	ccle_sample_info_with_tcga_label_imputation.to_csv(path.join("../data/1_filter_CCLE_with_TCGA",
	                                                             "CCLE_sample_info_with_imputed_TCGA_l"
	                                                             "abels_and_unified_site_names_step3_1248"
	                                                             ".csv"),
	                                                   index=False)
	"""
	solid tumours
	solid<-c("HNSC", "ESCA", "BRCA", "COAD/READ", "LIHC", "STAD", "ACC", "KIRC", "LUAD", "LUSC", "SCLC", "MESO", "GBM", "LGG", "MB", "NB", "PAAD", "SKCM", "THCA", "BLCA", "CESC", "UCEC", "OV", "PRAD")

	non-solid tumours
	liquid<-c("LAML", "DLBC", "LCML", "ALL", "MM", "CLL")
	used this annotation for one of my projects """


def set_ele_element_type_during_training(ft, source, tcga_int):
	ft = tf.cast(ft, tf.float32)
	source_label = tf.cast(source, tf.int32)  # given the label
	tcga_int = tf.cast(tcga_int, tf.int32)  # given the label
	return ft, source_label, tcga_int


def set_ele_element_type_during_test(ft, source, disease, tcga_lb, impute_tcga_lb, sample_id, tcga_int_label):
	"""
	:return:
	"""
	ft = tf.cast(ft, tf.float32)
	source_label = tf.cast(source, tf.int32)  # given the label
	disease = tf.cast(disease, tf.string)  # given the label
	tcga_lb = tf.cast(tcga_lb, tf.string)  # given the label
	impute_tcga_lb = tf.cast(impute_tcga_lb, tf.string)
	sample_id = tf.cast(sample_id, tf.int32)
	tcga_int_label = tf.cast(tcga_int_label, tf.int32)
	return ft, source_label, disease, tcga_lb, impute_tcga_lb, sample_id, tcga_int_label



def create_dataset(element_tuple_ds, batch_size=128, train_or_test="train", ds_name="train", shuffle=True):
	"""

	:param data_combo:  list of data that need to be in the dataset. [features, label, sampleID] or [features, label]
	:param labels: array, int labels-not anymore. after mixup it should be one-hot. so let's change all labels to
	one-hot
	:param batch_size:
	:param train_or_test: training or testing, training: train and val only two elements in the dataset,
	but in testing: all datasets are tracking multiple labesl.
	:param ds_name:
	:param epochs:
	:return:
	"""
	
	def set_ele_element_type_during_test(*args):
		"""
		:return:
		"""
		args = list(args)
		args[0] = tf.cast(args[0], tf.float32)  # ft
		args[1] = tf.cast(args[1], tf.int32)  # source_label  # given the label
		args[2] = tf.cast(args[2], tf.string)  # disease  # given the label
		args[3] = tf.cast(args[3], tf.string)  # tcga_lb  # given the label
		args[4] = tf.cast(args[4], tf.string)  # impute_tcga_lb
		args[5] = tf.cast(args[5], tf.int32)  # sample_id
		args[6] = tf.cast(args[6], tf.int32)  # tcga_int_label
		args[7] = tf.cast(args[7], tf.float32)  # tumor_percent
		args[8] = tf.cast(args[8], tf.float32)  # normal_percent
		return args
	
	def set_ele_element_type_during_training(*args):
		args = list(args)
		args[0] = tf.cast(args[0], tf.float32)
		args[1] = tf.cast(args[1], tf.int32)  # given the label
		args[2] = tf.cast(args[2], tf.int32)  # given the label
		if len(args) == 5:  # original 3 + 2 newly added
			args[3] = tf.cast(args[3], tf.float32)  # tumor_percent
			args[4] = tf.cast(args[4], tf.float32)  # normal_percent
		return args
	
	# Define a mapping function that takes *args
	def mapping_function_train(*args):
		return set_ele_element_type_during_training(*args)
	
	def mapping_function_test(*args):
		return set_ele_element_type_during_test(*args)
	
	dataset = tf.data.Dataset.from_tensor_slices(element_tuple_ds)
	if train_or_test == "test":
		dataset.map(mapping_function_test)
	
	else:  # during training
		if ds_name != "test":
			dataset.map(mapping_function_train)
	if shuffle:
		dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).repeat(1)
	else:
		dataset = dataset.batch(batch_size).repeat(1)
	
	dataset = dataset.prefetch(2)
	
	return dataset


def make_save_dir(args):
	from datetime import datetime
	args.time_str = '{0:%Y-%m-%dT%H-%M-%S}-'.format(datetime.now())
	# "norm_period: bin32-win200-step50-scan-min_IED_len66-DS256"
	args.results_dir = path.join(args.SAVE_DIR_ROOT,
	                             args.time_str)  # ,args.data_mode, args.sample_mode, args.n_sample_2select
	if not path.exists(args.results_dir):
		makedirs(args.results_dir)
	
	args.model_save_dir = path.join(args.results_dir, "model")
	return args
