import yaml
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os import path, makedirs
import pylab
from textwrap import wrap


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def save_yaml(self, save_file_name):
        with open(save_file_name, 'w') as outfile:
            yaml.dump(self.__dict__, outfile, default_flow_style=False)

def load_parameters(filename):
    with open(filename) as f:
        ym_dicts = yaml.load(f, Loader=yaml.FullLoader)
        args = Struct(**ym_dicts)
    return args


def getmeta2trackdict():
 """
 This is the dict with all meta info need to tracking during training/testing. This should be synced
 and shared between data processing pipeline and BasicDataSet class
 The meta harmonizing involves to find the corresponding columns for these following keys
 :return:
 """
 meta2track_dict = {"source_labels": {"dtype": np.int32, "values": []},  # this should be automatically
                    # assigned when combining multiple datasets
                         "primary_site": {"dtype": str, "values": []},
                         "diagnosis_b4_impute": {"dtype": str, "values": []},
                         "diagnosis": {"dtype": str, "values": []},
                         "tumor_percent": {"dtype": np.float32, "values": []},
                         "normal_percent": {"dtype": np.float32, "values": []},
                         "sample_id": {"dtype": np.float32, "values": []},
                         "tumor_stage": {"dtype": str, "values": []},
                         }

 return meta2track_dict

def get_reduced_dimension_projection(features, vis_mode="pacmap", n_components=2, n_neighbors=30,
                                     method_specific={
                                             "pacmap": {"MN_ratio": 0.5, "FP_ratio": 3},
                                             "umap": {"min_dist": 0.5, "spread": 5}
                                     }):
    """
     "umap": {"min_dist": 0.1, "spread": 5}
    """
    if vis_mode.lower() == "tsne":
        from bhtsne import tsne
        projection = tsne(features, n_components, rand_seed=0)
    elif vis_mode.lower() == "mds":
        from sklearn.manifold import MDS
        vis_model = MDS(n_components=n_components, random_state=66)
        projection = vis_model.fit_transform(features)
    elif vis_mode.lower() == "umap":
        from umap import UMAP
        umap_params = method_specific["umap"]
        # Setting n_neighbors to "None" leads to a default choice shown below in "parameter" section
        vis_model = UMAP(
            random_state=66, n_neighbors=n_neighbors, min_dist=umap_params["min_dist"],
            spread=umap_params["spread"], n_components=n_components, n_jobs=10)
        projection = vis_model.fit_transform(features)
    elif vis_mode.lower() == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        projection = pca.fit_transform(features)
    elif vis_mode.lower() == "pacmap":
        from pacmap import PaCMAP
        # Setting n_neighbors to "None" leads to a default choice shown below in "parameter" section
        pacmap_params = method_specific["pacmap"]  # default: MN_ratio=0.5, FP_ratio=2.0
        vis_model = PaCMAP(
            n_components=n_components, n_neighbors=n_neighbors, MN_ratio=pacmap_params["MN_ratio"],
            FP_ratio=pacmap_params["FP_ratio"], \
            random_state=66)
        projection = vis_model.fit_transform(features, init="pca")
    elif vis_mode.lower() == "phate":
        from phate import PHATE
        # Setting n_neighbors to "None" leads to a default choice shown below in "parameter" section
        pacmap_params = method_specific["pacmap"]  # default: MN_ratio=0.5, FP_ratio=2.0
        vis_model = PHATE(
            n_components=n_components, knn=8,
            random_state=66)
        projection = vis_model.fit_transform(features)
    return projection



def encode_labels(df, column_name):
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(df[column_name])
    return encoded_labels, encoder.classes_

def visualize_data_with_meta(features, meta_data_df, meta2check=[], cmap="viridis", n_cols=None,
                             n_rows=None,
                             postfix="STAD", figsize=[8, 6], save_dir="./", vis_mode="phate"):


    def is_df_column_string(df, column_name):
        col_datatype = df[column_name].dropna().dtypes
        return col_datatype == str or col_datatype == np.object_

    projection = get_reduced_dimension_projection(
        features, vis_mode=vis_mode,
        n_components=2,
        n_neighbors=30)

    if not n_cols:
        n_cols = np.int32(np.ceil(np.sqrt(len(meta2check))))
        n_rows = np.int32(np.ceil(len(meta2check) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    # Flatten the axes if it's not already a 1D array (in the case of a single plot)
    if n_rows * n_cols == 1:
        axes = [axes]  # Convert single axis to a list to be consistent with axes.flatten()
    else:
        axes = axes.flatten()
    plt.suptitle(postfix)
    for i, col in enumerate(meta2check):
        ax = axes[i]

        if is_df_column_string(meta_data_df, col):
            # Encode labels for string columns
            encoded_labels, classes = encode_labels(meta_data_df, col)
            color_labels = encoded_labels
            non_nan_classes = [ele for jj, ele in enumerate(classes) if str(ele) != "nan"]
        else:
            # Use the column values directly for numeric columns
            color_labels = meta_data_df[col].values
            non_nan_classes = np.unique(color_labels[~np.isnan(color_labels)])

        non_nan_inds = np.where(meta_data_df[col].values.astype(str) != "nan")[0]

        # Plot all points in gray
        ax.scatter(projection[:, 0], projection[:, 1], color="gray", alpha=0.65, s=5)

        # Plot non-NaN points with color
        sc = ax.scatter(
                projection[non_nan_inds, 0], projection[non_nan_inds, 1],
                c=color_labels[non_nan_inds], s=5, cmap=cmap)
        ax.set_title(textwrap.fill(str(col), width=30), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        # Add colorbar to the current axes
        cbar = fig.colorbar(sc, ax=ax)

        # Set the ticks and labels for the colorbar
        if is_df_column_string(meta_data_df, col):
            cbar.set_ticks(np.arange(len(non_nan_classes)))
            wrapped_labels = [textwrap.fill(str(label), width=25) for label in non_nan_classes]
            cbar.set_ticklabels(wrapped_labels, fontsize=8)

    # Adjust layout for better fit
    plt.tight_layout()
    plt.savefig(path.join(save_dir, f"vis_meta_{postfix}.png"))
    plt.close()

    return projection


def generate_pastel_colors(num_colors):
    import random
    def get_random_color(pastel_factor=0.5):
        return [(x + pastel_factor) / (1.0 + pastel_factor) for x in [random.uniform(0, 1.0) for i in [1, 2, 3]]]

    def color_distance(c1, c2):
        return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])

    def generate_new_color(existing_colors, pastel_factor=0.5):
        max_distance = None
        best_color = None
        for i in range(0, 100):
            color = get_random_color(pastel_factor=pastel_factor)
            if not existing_colors:
                return color
            best_distance = min([color_distance(color, c) for c in existing_colors])
            if not max_distance or best_distance > max_distance:
                max_distance = best_distance
                best_color = color
        return best_color

    colors = []
    for i in range(0, num_colors):
        new_color = generate_new_color(colors, pastel_factor=0.9)
        colors.append(new_color)

    return colors

"""Bokeh related plotting functions start"""
# 2024.05.31
def interactive_bokeh_with_select(x, y,
                                  hover_notions=[("pat_ids", np.arange(10))],
                                  color_by="label", height=500, width=500,
                                  title="Title", mode="tsne",
                                  table_columns=["x", "y"],
                                  s2_hoverkeys=["genes", "index"],
                                  postfix="postfix",
                                  if_color_gradient=False,
                                  save_dir="../results", scatter_size=3):
    """
        # Great! https://github.com/surfaceowl-ai/python_visualizations/blob/main/notebooks
        /bokeh_save_linked_plot_data.ipynb
            # Generate linked plots + TABLE displaying data + save button to export cvs of
            selected data
        :param x: array of the projection[:, 0], x locations in the pacmap plot
        :param y: projection[:, 0], y locations in the pacmap plot
        :param indiv_id:
        :param colormap: bokeh palettes
        :param hover_notions:
        :param cmap_interval:
        :param xlabel:
        :param ylabel:
        :param title:
        :param mode:
        :param plot_func:
        :param postfix:
        :param save_dir:
        :return:
        """
    from bokeh.layouts import grid
    from bokeh.plotting import figure, output_file
    from bokeh.plotting import save as bokeh_save

    # Define the callback function to sort the table by a given column
    if not path.exists(save_dir):
        makedirs(save_dir)

    data_dict = {"x": x, "y": y}
    plot_width, plot_height = height, width

    tooltips = []
    for key, value in hover_notions:
        data_dict.update({key: value})
        tooltips.append((key, "@{}".format(key)))

    # only plot non-nan values
    ## compute color for this column
    has_nan, nan_inds, number2uniq, uniq_values, used_color_palette = get_unique_assign_color(
            data_dict, if_color_gradient=if_color_gradient, color_key="colors_bk",
            key2color=color_by
    )

    if has_nan:  # set nan values to gray
        for ind in nan_inds:
            data_dict["colors_bk"][ind] = "gray"
    cmap_interval = len(uniq_values) - 1  # max value for the color map

    fig01, fig_select, table, savebutton = (
            bokeh_plot_main_with_select_table(
                    data_dict,
                    cmap_interval=cmap_interval, number2uniq=number2uniq,
                    plot_height=plot_height, plot_width=plot_width, s2_hoverkeys=s2_hoverkeys,
                    scatter_size=scatter_size, table_columns=table_columns,
                    title=title,
                    tooltips=tooltips, ticks_loc=np.linspace(
                            0, cmap_interval, len(uniq_values)),
                    used_color_palette=used_color_palette))

    # display results
    layout2 = grid([fig01, fig_select, table, savebutton], ncols=4)
    output_file(save_dir + '/Proj-on-{}.html'.format(postfix), title=title)
    bokeh_save(layout2)
    # bokeh_show(layout2)
    print(f"{color_by} is done!")

def get_unique_assign_color(data_dict, if_color_gradient=False, color_key="colors_bk",
                            key2color="diagnosis"):
    has_nan, nan_inds, non_nan_values, uniq_values = get_uniq_values_dealing_nan(
            data_dict, color_by=key2color
    )
    # Create a mapping from unique values to indices
    uniq2number = {str(value): idx for idx, value in enumerate(uniq_values)}
    number2uniq = {val: "\n".join(wrap(str(key), 10)) for key, val in uniq2number.items()}
    # Encode values
    reencode_values = np.array([uniq2number[str(val)] for val in data_dict[key2color]])
    # Assign colors to non-nan values and set the used color palette
    data_dict[color_key], used_color_palette = assign_colors(
            uniq_values, reencode_values,
            if_color_gradient
    )

    if has_nan:  # set nan values to gray
        for ind in nan_inds:
            data_dict["colors_bk"][ind] = "#ededed"
    return has_nan, nan_inds, number2uniq, uniq_values, used_color_palette


def assign_colors(values, reencode_values, if_color_gradient):
    if if_color_gradient:
        cmap_colors = pylab.cm.viridis(np.linspace(0, 1, len(values)))
        colors = [f"#{int(r):02x}{int(g):02x}{int(b):02x}"
                  for r, g, b, _ in 255 * cmap_colors[reencode_values]]
        used_color_palette = "Viridis256"
    else:
        generated_colors = generate_pastel_colors(len(values))
        colors = [f"#{int(r):02x}{int(g):02x}{int(b):02x}"
                  for r, g, b in 255 * np.array(generated_colors)[reencode_values]]
        used_color_palette = tuple(
                [f"#{int(r):02x}{int(g):02x}{int(b):02x}"
                 for r, g, b in 255 * np.array(generated_colors)])
    return colors, used_color_palette


def get_uniq_values_dealing_nan(data_dict, color_by="colorby"):
    # only plot non-nan values
    nan_inds = [ind for ind, ele in enumerate(data_dict[color_by]) if str(ele) == "nan"]
    non_nan_inds = [ind for ind, ele in enumerate(data_dict[color_by]) if str(ele) != "nan"]
    has_nan = len(nan_inds) > 0

    # Extract non-NaN values
    non_nan_values = np.array(data_dict[color_by])[non_nan_inds]

    # Get unique values of non-NaN entries, handling potential errors
    try:
        uniq_values_nonan = np.unique(non_nan_values)
    except:
        uniq_values_nonan = np.unique(non_nan_values.astype(str))

    # If there are NaN values, we need to account for them
    if has_nan:
        uniq_values, color_by_values = process_and_sort_unique_values(
            non_nan_values, uniq_values_nonan)
        uniq_values = ["nan"] + list(uniq_values)
        data_dict[color_by][non_nan_inds] = color_by_values
    else:
        uniq_values, color_by_values = process_and_sort_unique_values(
            non_nan_values, uniq_values_nonan)
        data_dict[color_by][non_nan_inds] = color_by_values
    return has_nan, nan_inds, non_nan_values, uniq_values

def process_and_sort_unique_values(values, uniq_values):
    """
    Process and sort unique values, handling numeric types and rounding where necessary.

    Parameters:
    - values: Array of values to process
    - uniq_values: Array of unique values

    Returns:
    - Sorted unique values
    """
    if len(uniq_values) > 0:
        if isinstance(uniq_values[0], (int, float)):
            if isinstance(uniq_values[0], int):
                values = np.array(values).astype(np.int32)
            else:
                values = np.array(values).round(3)
            uniq_values = np.unique(values)

    uniq_values.sort()
    return uniq_values, values



# 2024.05.31
def interactive_bokeh_all_meta_in1_subplots(
        x, y, meta_df,
        hover_notions=[("pat_ids", np.arange(10))],
        height=500, width=500,
        title="Title", mode="tsne",
        table_columns=["x", "y"],
        s2_hoverkeys=["genes", "index"],
        key2color="diagnosis",
        postfix="postfix",
        if_color_gradient=False,
        if_indiv_proj=False,
        num_per_fig=50,
        save_dir="../results", scatter_size=3
):
    """
    This plot all the cols in the given meta_df with the given x, y for vis.
        # Great! https://github.com/surfaceowl-ai/python_visualizations/blob/main/notebooks
        /bokeh_save_linked_plot_data.ipynb
            # Generate linked plots + TABLE displaying data + save button to export cvs of
            selected data
        :param x: array of the projection[:, 0], x locations in the pacmap plot
        :param y: projection[:, 0], y locations in the pacmap plot
        :param indiv_id:
        :param colormap: bokeh palettes
        :param hover_notions:
        :param cmap_interval:
        :param xlabel:
        :param ylabel:
        :param title:
        :param mode:
        :param plot_func:
        :param postfix:
        :param save_dir:
        :return:
        """
    from datetime import datetime
    from bokeh.layouts import row, grid
    from bokeh.models import CustomJS, ColumnDataSource, HoverTool, Button, ColorBar, FixedTicker
    from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
    from bokeh.plotting import figure, output_file
    from bokeh.plotting import show, save

    if not path.exists(save_dir):
        makedirs(save_dir)

    title = f"{title}"
    data_dict = {"x": x, "y": y}
    plot_width, plot_height = height, width

    numfigs = len(meta_df.columns) // num_per_fig + 1 if np.mod(
            len(meta_df.columns), num_per_fig) != 0 else len(meta_df.columns) // num_per_fig
    for fig in range(numfigs):
        curent_fig_rec = pd.DataFrame(
                np.zeros((num_per_fig, 2)), columns=[f"fig{fig}_cols", "nuniq_gt_2"])
        subfigs = []
        for jj, col in enumerate(list(meta_df.columns)[fig * num_per_fig: (fig + 1) * num_per_fig]):
            curent_fig_rec.at[
                jj, f"fig{fig}_cols"] = col  # this df track which meta col is in which fig
            curent_fig_rec.at[jj, "nuniq_gt_2"] = False
            sub_hover_notions = hover_notions
            if np.array([col not in ele[0] for ele in hover_notions]).all():
                sub_hover_notions.append((col, meta_df[key]))

            tooltips = []
            for key, value in sub_hover_notions:
                data_dict.update({key: value})
                tooltips.append((key, "@{}".format(key)))

            ## compute color for this column
            has_nan, nan_inds, number2uniq, uniq_values, used_color_palette = (
                    get_unique_assign_color(
                    data_dict, if_color_gradient=if_color_gradient, color_key="colors_bk",
                    key2color=col
            ))
            cmap_interval = len(uniq_values) - 1  # defines the colorbar apprearance
            if len(uniq_values) >= 2 and len(uniq_values) < meta_df.shape[0] // 2:
                print(f"{jj} {col}: {uniq_values}")
                curent_fig_rec.at[jj, "nuniq_gt_2"] = True
                # set the selected-fig hoverkeys
                s2_hoverkeys = list(np.unique(s2_hoverkeys + [col]))

                # create dynamic table of selected points
                columns = []
                current_table_columns = list(set(table_columns + [col]))
                for c in current_table_columns:
                    columns.append(TableColumn(field=f"{c}", title=f"{c}"))

                fig01, fig_select, table, savebutton = bokeh_plot_main_with_select_table(
                        data_dict,
                        cmap_interval=cmap_interval, number2uniq=number2uniq,
                        plot_height=plot_height, plot_width=plot_width, s2_hoverkeys=s2_hoverkeys,
                        scatter_size=scatter_size, table_columns=current_table_columns, title=col,
                        tooltips=tooltips, ticks_loc=np.linspace(
                                0, cmap_interval, len(uniq_values)),
                        used_color_palette=used_color_palette)
                subfigs += [fig01, fig_select, table, savebutton]
        layout2 = grid(subfigs, ncols=4)
        output_file(
                save_dir + f'/fig{fig}-all{len(meta_df.columns)}-Meta_in1_Proj-on-'
                           f'{postfix}-subfigs.html',
                title=title)
        save(layout2)

        curent_fig_rec.to_csv(path.join(save_dir, f"fig{fig}_meta_col.csv"))
        print("ok")

def interactive_bokeh_with_select_test_for_DGEA(x, y,
                                                hover_notions=[("pat_ids", np.arange(10))],
                                                key2separate="disease_code",
                                                key2color="binary_response",
                                                height=500, width=500,
                                                title="Title", mode="tsne",
                                                table_columns=["x", "y"],
                                                s2_hoverkeys=["genes", "index"],
                                                postfix="postfix",
                                                default_save_name=None,
                                                if_color_gradient=False,
                                                if_multi_separate=False,
                                                save_dir="../results", scatter_size=3):

    """
        # Great! https://github.com/surfaceowl-ai/python_visualizations/blob/main/notebooks
        /bokeh_save_linked_plot_data.ipynb
            # Generate linked plots + TABLE displaying data + save button to export cvs of
            selected data
        :param x: array of the projection[:, 0], x locations in the pacmap plot
        :param y: projection[:, 0], y locations in the pacmap plot
        :param indiv_id:
        :param colormap: bokeh palettes
        :param hover_notions:
        :param cmap_interval:
        :param xlabel:
        :param ylabel:
        :param title:
        :param mode:
        :param plot_func:
        :param postfix:
        :param save_dir:
        :return:
        """
    # Define the callback function to sort the table by a given column
    if not path.exists(save_dir):
        makedirs(save_dir)

    data_dict = {"x": x, "y": y}
    plot_width, plot_height = height, width

    tooltips = []
    for key, value in hover_notions:
        data_dict.update({key: value})
        tooltips.append((key, "@{}".format(key)))

    # only plot non-nan values
    ## compute color for this column
    has_nan, nan_inds, number2uniq, uniq_values, used_color_palette = get_unique_assign_color(
            data_dict, if_color_gradient=if_color_gradient, color_key="colors_bk",
            key2color=key2color
    )

    if has_nan:  # set nan values to gray
        for ind in nan_inds:
            data_dict["colors_bk"][ind] = "gray"
    cmap_interval = len(uniq_values) - 1  # max value for the color map

    bokeh_plot_main_with_2select_for_DGEA(
            data_dict, cmap_interval=cmap_interval, number2uniq=number2uniq,
            plot_height=plot_height, plot_width=plot_width,
            s2_hoverkeys=s2_hoverkeys, scatter_size=5,
            table_columns=table_columns, title=title, tooltips=tooltips,
            postfix=postfix, default_save_name=default_save_name,
            if_multi_separate=if_multi_separate,
            key2separate=key2separate, key2color=key2color,
            ticks_loc=np.linspace(
                    0, cmap_interval, len(uniq_values)),
            used_color_palette=used_color_palette, save_dir=save_dir)

    print(f"{key2color} is done!")

def bokeh_plot_main_with_2select_for_DGEA(data_dict, cmap_interval=1, number2uniq={0: "ACC"},
                                          plot_height=500, key2separate="disease_code",
                                          key2color="binary_response",
                                          plot_width=500, s2_hoverkeys=["x", "y"], scatter_size=5,
                                          table_columns=["x", "y"], title="main title", tooltips=[],
                                          postfix="postfix", default_save_name=None,
                                          if_multi_separate=True,
                                          ticks_loc=np.linspace(0, 1, 10),
                                          used_color_palette={"a": "gray"}, save_dir="./"):
    """
    This function, provide interactive differential gene expression analysis
    Make sure in data dict and gene_features_df have shared sample identifier that can be used
    for DGEA.
    :param data_dict:
    :param gene_features_df:
    :param cmap_interval:
    :param number2uniq:
    :param plot_height:
    :param plot_width:
    :param s2_hoverkeys:
    :param scatter_size:
    :param table_columns:
    :param title:
    :param tooltips:
    :param ticks_loc:
    :param used_color_palette:
    :return:
    """
    from bokeh.plotting import figure, output_file, show, save
    from bokeh.plotting import save as bokeh_save

    overall_layout = []
    layout = bokeh_plot_for_subsetAB_multiple_unified_color(
            data_dict, cmap_interval=cmap_interval,
            number2uniq=number2uniq, plot_height=plot_height, plot_width=plot_width,
            s2_hoverkeys=s2_hoverkeys, save_dir=save_dir, scatter_size=scatter_size,
            table_columns=table_columns, annotationA="A", annotationB="B",
            ticks_loc=ticks_loc, title=f"{title}-colorby-{key2color}", tooltips=tooltips,
            used_color_palette=used_color_palette)
    overall_layout.append(layout)

    if if_multi_separate:
        str_data = np.array(data_dict[key2separate]).astype(str)
        uniq_key2separate = np.unique(str_data)

        sub_data_dict = {}
        for q in uniq_key2separate:
            q_inds = np.where(str_data == q)[0]

            for key in data_dict.keys():
                sub_data_dict[key] = np.array(data_dict[key])[q_inds]

            if len(q_inds) > 5:
                layout = bokeh_plot_for_subsetAB_multiple_unified_color(
                        sub_data_dict, cmap_interval=cmap_interval,
                        number2uniq=number2uniq, plot_height=plot_height, plot_width=plot_width,
                        s2_hoverkeys=s2_hoverkeys, save_dir=save_dir, scatter_size=scatter_size,
                        table_columns=table_columns, annotationA=f"{title}-{q}", annotationB=f"{q}",
                        ticks_loc=ticks_loc, title=f"{q} (n={len(q_inds)})",
                        default_save_name=default_save_name, tooltips=tooltips,
                        used_color_palette=used_color_palette)
                overall_layout.append(layout)
    # Output file
    output_file(path.join(save_dir, f'{postfix}-for-AB.html'))
    bokeh_save(overall_layout)


def bokeh_plot_for_subsetAB_multiple_unified_color(data_dict, cmap_interval=1,
                                                   number2uniq={0: "ACC"},
                                                   plot_height=500,
                                                   sample_identifier_col="short_sample_id",
                                                   plot_width=500, s2_hoverkeys=["x", "y"],
                                                   scatter_size=5,
                                                   table_columns=["x", "y"], title="main title",
                                                   default_save_name=None, tooltips=[],
                                                   postfix="postfix", annotationA="A",
                                                   annotationB="B",
                                                   ticks_loc=np.linspace(0, 1, 10),
                                                   used_color_palette={"a": "gray"}, save_dir="./"):

    from bokeh.models import (Button, CustomJS, ColumnDataSource, HoverTool, TableColumn,
                              DataTable, \
        Div, TextInput)
    from bokeh.layouts import column, row
    from bokeh.plotting import figure
    from datetime import datetime

    def get_subsetAB_plots(plot_height, plot_width, s2_hoverkeys, sA, sB, scatter_size):
        # Create Selection A subplot
        figA = figure(
                width=plot_width // 2, height=plot_height // 2,
                tools=["box_zoom", 'pan', "wheel_zoom", "reset", "save"], title="Selection A")
        figA.scatter(
                'x', 'y', size=scatter_size + 3, source=sA, alpha=0.8, fill_color='colors_bk',
                line_color='colors_bk')
        # Create Selection B subplot
        figB = figure(
                width=plot_width // 2, height=plot_height // 2,
                tools=["box_zoom", 'pan', "wheel_zoom", "reset", "save"], title="Selection B")
        figB.scatter(
                'x', 'y', size=scatter_size + 3, source=sB, alpha=0.8, fill_color='colors_bk',
                line_color='colors_bk')
        # Add Hover tools for A and B selections
        figA.add_tools(HoverTool(tooltips=[(key, f"@{key}") for key in s2_hoverkeys]))
        figB.add_tools(HoverTool(tooltips=[(key, f"@{key}") for key in s2_hoverkeys]))
        return figA, figB

    def get_buttons_setAB(s1, sA, sB):
        active_selection = TextInput(value="A", visible=False)  # Default to A
        # Set A button callback: Set active selection to "A"
        setA_button = Button(label="Set A", button_type="primary")
        setA_button.js_on_click(
                CustomJS(
                        args=dict(active_selection=active_selection), code="""
            active_selection.value = 'A';
        """))
        # Set B button callback: Set active selection to "B"
        setB_button = Button(label="Set B", button_type="primary")
        setB_button.js_on_click(
                CustomJS(
                        args=dict(active_selection=active_selection), code="""
            active_selection.value = 'B';
        """))
        # CustomJS code for updating the selection
        code_update_selection = """
            var inds = cb_obj.indices;
            var d1 = s1.data;
            var d2 = active_selection.value === 'A' ? sA.data : sB.data;
            for (var key in d1) {
                d2[key] = [];
            }
            for (var i = 0; i < inds.length; i++) {
                for (var key in d1) {
                    d2[key].push(d1[key][inds[i]]);
                }
            }
            if (active_selection.value === 'A') {
                sA.change.emit();
            } else {
                sB.change.emit();
            }
        """
        # Assign JS callback for selection (will update either sA or sB based on the active
        # selection stored in TextInput)
        s1.selected.js_on_change(
                "indices", CustomJS(
                        args=dict(s1=s1, sA=sA, sB=sB, active_selection=active_selection),
                        code=code_update_selection))
        return active_selection, setA_button, setB_button

    def get_download_save_buttonAB(sA, sB, text_input_A, text_input_B, filename_input):
        """"""
        # Button to download the selected data
        download_button = Button(label="Download Selected Data", button_type="success")
        # JavaScript code to download data from sA and sB
        download_callback = CustomJS(
                args=dict(
                        sA=sA, sB=sB, filename_input=filename_input, text_input_A=text_input_A,
                        text_input_B=text_input_B), code="""
            function download(filename, text) {
                var element = document.createElement('a');
                element.setAttribute('href', 'data:text/csv;charset=utf-8,' + encodeURIComponent(
                text));
                element.setAttribute('download', filename);
                element.style.display = 'none';
                document.body.appendChild(element);
                element.click();
                document.body.removeChild(element);
            }

            var annotationA = text_input_A.value
            var annotationB = text_input_B.value
            // Get the sanitized filename from TextInput
            var sanitized_filename = filename_input.value;

            // Append annotations to the filename
            sanitized_filename += "-" + annotationA + "-" + annotationB;

            // Split the full file path into directory and file parts
            var path_parts = sanitized_filename.split('|');
            var directory = path_parts.slice(0, -1).join('/');  // Reconstruct the directory 
            structure with slashes
            var filename = path_parts.slice(-1)[0];  // Get the last part as the filename

            // Construct the final filename with directory intact and sanitized filename
            var final_filename = directory + "/" + filename + ".csv";

            // Create CSV content
            var dataA = sA.data;
            var dataB = sB.data;
            var columns = Object.keys(dataA);  // Get all column names in dataA

            var csvContent = "Selection," + columns.join(",") + "\\n";  // Create CSV header with 
            column names

            // Loop through dataA
            for (var i = 0; i < dataA[columns[0]].length; i++) {  // Loop over rows
                var row = "A";  // Start with the selection identifier
                for (var j = 0; j < columns.length; j++) {  // Loop over columns
                    row += "," + dataA[columns[j]][i];  // Append each column value for the 
                    current row
                }
                csvContent += row + "\\n";  // Add the row to the CSV content
            }

            // Loop through dataB
            for (var i = 0; i < dataB[columns[0]].length; i++) {  // Loop over rows
                var row = "B";  // Start with the selection identifier
                for (var j = 0; j < columns.length; j++) {  // Loop over columns
                    row += "," + dataB[columns[j]][i];  // Append each column value for the 
                    current row
                }
                csvContent += row + "\\n";  // Add the row to the CSV content
            }

            // Trigger the download with the full filename
            download(sanitized_filename, csvContent);
        """)
        # Assign the JavaScript callback to the button
        download_button.js_on_click(download_callback)
        #
        # saved_filename = "selected_data_" + timestamp + ".csv"
        return download_button

    fig01, s1 = func_bokeh_plot_scatter_points_fig1(
            data_dict, number2uniq, tooltips,
            ticks_loc=ticks_loc,
            plot_height=plot_height, plot_width=plot_width,
            cmap_interval=cmap_interval,
            used_color_palette=used_color_palette,
            scatter_size=scatter_size, title=title
    )
    # Create two empty ColumnDataSources for Selection A and B
    data_dict_default = {key: [] for key in list(s2_hoverkeys) + ["x", "y"]}
    sA = ColumnDataSource(data=data_dict_default)  # Selection A data
    sB = ColumnDataSource(data=data_dict_default)  # Selection B data

    ## plot subset A, B into two figures
    figA, figB = get_subsetAB_plots(plot_height, plot_width, s2_hoverkeys, sA, sB, scatter_size)
    # Create a dynamic table for selected A, B points
    columns = [TableColumn(field=f"{col}", title=f"{col}") for col in table_columns]
    tableA = DataTable(
            source=sA, columns=columns, width=plot_width, height=plot_height // 2, sortable=True,
            selectable=True, editable=True)
    tableB = DataTable(
            source=sB, columns=columns, width=plot_width, height=plot_height // 2, sortable=True,
            selectable=True, editable=True)

    # Add "Set A" and "Set B" buttons to determine which selection to update
    active_selection, setA_button, setB_button = get_buttons_setAB(s1, sA, sB)
    # Create input fields for annotations (names) for Selection A and B
    text_input_A = TextInput(value=annotationA, title="Enter name for Selection A")
    text_input_B = TextInput(value=annotationB, title="Enter name for Selection B")

    # Create TextInput for filename input
    if default_save_name is None:
        timestamp = datetime.now().strftime("%m-%dT%H-%M-%S")
    else:
        timestamp = default_save_name
    # Full file path to be used in the backend for saving, with annotations integrated
    base_filename = f"{timestamp}"
    filename_input = TextInput(
            value=f"{base_filename}",
            title=f"Enter save file name (no extension)\npath: {save_dir}", width=800)
    ## add annotation A, B in the saved csv
    download_button = get_download_save_buttonAB(sA, sB, text_input_A, text_input_B, filename_input)

    # Layout for the app
    layout = column(
            row(
                    fig01, column(setA_button, setB_button), column(figA, figB),
                    column(tableA, tableB)),
            row(text_input_A, text_input_B, filename_input),
            download_button, active_selection  # Hidden input field to track the active selection
    )
    return layout


def bokeh_plot_main_with_select_table(sub_data_dict, cmap_interval=1, number2uniq={0: "ACC"},
                                      plot_height=500,
                                      plot_width=500, s2_hoverkeys=["x", "y"], scatter_size=5,
                                      table_columns=["x", "y"], title="main title", tooltips=[],
                                      ticks_loc=np.linspace(0, 1, 10),
                                      used_color_palette={"a": "gray"}):
    from datetime import datetime
    from bokeh.models import CustomJS, ColumnDataSource, HoverTool, Button, ColorBar, FixedTicker
    from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
    fig01, s1 = func_bokeh_plot_scatter_points_fig1(
            sub_data_dict, number2uniq, tooltips,
            ticks_loc=ticks_loc,
            plot_height=plot_height, plot_width=plot_width,
            cmap_interval=cmap_interval,
            used_color_palette=used_color_palette,
            scatter_size=scatter_size, title=title
    )
    ## plot selected points
    s2_hoverkeys = s2_hoverkeys + ["colors_bk"]

    code, fig_select, s2 = func_bokeh_select_subset_from_fig(
            sub_data_dict, scatter_size=scatter_size + 3,
            plot_height=plot_height - 100,
            plot_width=plot_width - 100,
            hoverkeys=s2_hoverkeys, title=""
    )
    # create dynamic table of selected points
    columns = []
    for col in table_columns:
        columns.append(TableColumn(field=f"{col}", title=f"{col}"))
    table = DataTable(
            source=s2, columns=columns, width=plot_width, height=plot_height,
            sortable=True,
            selectable=True,
            editable=True
    )
    # add selected subset of f1 to s2
    s1.selected.js_on_change(
            "indices", CustomJS(args=dict(s1=s1, s2=s2, table=table), code=code)
    )
    timestamp = datetime.now().strftime("%d-%HT%M-%S")
    savebutton = bokeh_save_button(
            s1, table_columns=table_columns,
            save_name=f"{title}-{timestamp}.csv"
    )
    return fig01, fig_select, table, savebutton


def func_bokeh_plot_scatter_points_fig1(data_dict, tick_name_dict, tooltips,
                                        ticks_loc=[0.5, 1.5, 2.5],
                                        plot_height=500, plot_width=500, xlabel="embedding #1",
                                        y_label="embedding #2",
                                        cmap_interval=1, title="title",
                                        used_color_palette="Viridis", scatter_size=2):
    """
    plot figure one with data
    :param data_dict: dict {x:.., y:.., keys:..}
    :param tick_name_dict: customize colorbar tick labels
    :param ticks: colorbar ticks, np.ceil(np.linspace(max(0, np.min(reencode_values)),
    cmap_interval,
    min(len(reencode_values), 5))).astype(np.int32)
    :param plot_height:
    :param plot_width:
    :param postfix:
    :param mode: tsne, pca, pacmap
    :param cmap_interval: colormap max-value
    :param used_color_palette: color palette
    :param scatter_size:
    :return:
    """
    from bokeh.models import ColumnDataSource, HoverTool, ColorBar, FixedTicker, LinearColorMapper
    from bokeh.plotting import figure
    from bokeh.transform import linear_cmap

    s1 = ColumnDataSource(data=data_dict)
    # hovers = HoverTool(renderers=["train"], tooltips=tooltips)
    fig01 = figure(
        width=plot_width, height=plot_height,
        tools=['box_select', 'pan', "box_zoom", 'wheel_zoom', "lasso_select", "hover", "undo",
               'reset'],  # hovers,'pan', "box_zoom",

        title=title)
    fig01.xaxis.axis_label = xlabel
    fig01.yaxis.axis_label = y_label
    fig01.scatter(
        'x', 'y', size=scatter_size, alpha=0.9, line_width=1, fill_color='colors_bk',
        line_color='colors_bk',
        source=s1, name="train")
    mapper = linear_cmap(
        field_name='time', palette=used_color_palette, low=0,
        high=cmap_interval)
    ## color and label mismatch, pay attention to the high and low of the colorbar!
    color_ticks = FixedTicker(ticks=ticks_loc)
    color_bar = ColorBar(
        color_mapper=mapper['transform'], width=12, location=(0, 0),
        ticker=color_ticks, major_label_overrides=tick_name_dict)  #
    fig01.add_layout(color_bar, 'right')
    fig01.add_tools(HoverTool(tooltips=tooltips))
    return fig01, s1

def bokeh_save_button(source_data, table_columns=["diagnosis", "index"], save_name="save_name,txt"):
    """
    # create save button - saves selected datapoints to text file onbutton
    # inspriation for this code:
    # credit:  https://stackoverflow.com/questions/31824124/is-there-a-way-to-save-bokeh-data
    -table-content
    # note: savebutton line `var out = "x, y\\n";` defines the header of the exported file,
    helpful to have a header for downstream processing

    :param color_by: str, key
    :param source_data: dict
    :param table_columns: list, of columns to save
    :return:
    """
    from bokeh.models import CustomJS, Button
    from bokeh.events import ButtonClick  # for saving data
    savebutton = Button(label="Save", button_type="success")

    stringll = ""
    for ele in table_columns[:-1]:
        stringll += f'{ele}, '
    out = f'"{stringll}{table_columns[-1]}\\n"'
    code = f"""
            var inds = source_data.selected.indices;
            var data = source_data.data;
            var out = {out};\n"""
    code += "for (var i = 0; i < inds.length; i++) {\n"
    code += "out+="
    for key in table_columns:
        code += f"data['{key}'][inds[i]] + \",\" +"
    code = code[:-5]  # remove last ","
    code += '"\\n"; \n'
    code += f"""
    }}
    var file = new Blob([out], {{type: 'text/plain'}});
    var elem = window.document.createElement('a');
    elem.href = window.URL.createObjectURL(file);
    elem.download = '{save_name}';
    document.body.appendChild(elem);
    elem.click();
    document.body.removeChild(elem);
    """

    savebutton.js_on_event(ButtonClick, CustomJS(args=dict(source_data=source_data), code=code))
    return savebutton

def func_bokeh_select_subset_from_fig(data_dict, scatter_size=5, plot_height=600, plot_width=600,
                                      hoverkeys=["diagnosis", "index"], title="title",
                                      plot_mode="scatter"):
    """
    # inspiration for this from a few sources:
    https://stackoverflow.com/questions/54215667/bokeh-click-button-to-save-widget-values-to-txt
    -file-using-javascript
    # credit: https://stackoverflow.com/users/1097752/iolsmit
    via: https://stackoverflow.com/questions/48982260/bokeh-lasso-select-to-table-update
    # credit: https://stackoverflow.com/users/8412027/joris
    via: https://stackoverflow.com/questions/34164587/get-selected-data-contained-within-box
    -select-tool-in-bokeh
    Add f2 for xoomed-in selection from f1
    :param data_dict:
    :param scatter_size:
    :return:
    """

    from bokeh.models import ColumnDataSource, HoverTool, ColorBar, FixedTicker, LinearColorMapper
    from bokeh.plotting import figure

    data_dict_default = {key: [] for key in list(hoverkeys) + ["x", "y"]}
    s2 = ColumnDataSource(data=data_dict_default)
    # demo smart error msg:  `box_zoom`, vs `BoxZoomTool`
    fig02 = figure(
        width=plot_width, height=plot_height, tools=["box_zoom", 'pan', "wheel_zoom", "reset",
                                                     "save"],
        title=title)  # x_range=(0, 1), y_range=(0, 1),
    if plot_mode == "scatter":
        fig02.scatter(
            "x", "y", size=scatter_size, source=s2, alpha=0.8, fill_color='colors_bk',
            line_color='colors_bk')  #
    elif plot_mode == "line":
        fig02.line('x', 'y', source=s2)
        fig02.scatter(
            'x', 'y', source=s2, size=scatter_size, alpha=0.8, fill_color='colors_bk',
            line_color='colors_bk')  #
    # fancy javascript to link subplots
    # js pushes selected points into ColumnDataSource of 2nd plot

    code = """
        var inds = cb_obj.indices;
        var d1 = s1.data;
        var d2 = s2.data;"""
    for key in data_dict.keys():
        code += f"d2['{key}'] = []; \n"
    code += "for (var i = 0; i < inds.length; i++) {\n"
    for key in data_dict.keys():
        code += f"d2['{key}'].push(d1['{key}'][inds[i]]);\n"
    code += """
        }
        s2.change.emit();
        table.change.emit();
        var inds = source_data.selected.indices;
        var data = source_data.data;
        var out = "x, y\\n";
        for (i = 0; i < inds.length; i++) {
        """
    code += "out+="
    for key in data_dict.keys():
        code += f"data['{key}'][inds[i]] + \",\" +"
    code = code[:-5]  # remove last ","
    code += '"\\n"; \n'
    code += """
        }
        var file = new Blob([out], {type: 'text/plain'});
         """

    # add Hover tool
    # define what is displayed in the tooltip
    tooltips = []
    if "colors_bk" in hoverkeys:
        hoverkeys.remove("colors_bk")
    for key in hoverkeys:
        tooltips.append((key, f"@{key}"))

    fig02.add_tools(HoverTool(tooltips=tooltips))

    return code, fig02, s2



# 2024.05.31
def interactive_bokeh_multi_subplots_unify_colorcoding(
        x, y, features,
        hover_notions=[("pat_ids", np.arange(10))],
        height=500, width=500,
        title="Title",
        table_columns=["x", "y"],
        s2_hoverkeys=["genes", "index"],
        key2color="diagnosis",
        key2separate="diagnosis",
        postfix="postfix",
        if_color_gradient=False,
        if_indiv_proj=False,
        save_dir="../results", scatter_size=3
):
    """
        # Great! https://github.com/surfaceowl-ai/python_visualizations/blob/main/notebooks
        /bokeh_save_linked_plot_data.ipynb
            # Generate linked plots + TABLE displaying data + save button to export cvs of
            selected data
        :param x: array of the projection[:, 0], x locations in the pacmap plot
        :param y: projection[:, 0], y locations in the pacmap plot
        :param indiv_id:
        :param colormap: bokeh palettes
        :param hover_notions:
        :param cmap_interval:
        :param xlabel:
        :param ylabel:
        :param title:
        :param mode:
        :param plot_func:
        :param postfix:
        :param save_dir:
        :return:
        """
    from bokeh.plotting import figure, output_file
    from bokeh.layouts import row, grid
    from bokeh.plotting import save as bokeh_save

    # Define the callback function to sort the table by a given column
    def sort_table(source, table, column_name):
        sorted_indices = source.data[column_name].argsort()
        source.data = {key: [value[i] for i in sorted_indices] for key, value in
                       source.data.items()}
        table.source = source  # Update the source of the DataTable

    if not path.exists(save_dir):
        makedirs(save_dir)

    title = f"{title}-per{key2separate}"
    data_dict = {"x": x, "y": y}
    plot_width, plot_height = height, width

    tooltips = []
    for key, value in hover_notions:
        data_dict.update({key: value})
        tooltips.append((key, "@{}".format(key)))

    has_nan = {f"{key2separate}": 0, f"{key2color}": 0}
    nan_inds = {f"{key2separate}": 0, f"{key2color}": 0}
    non_nan_values = {f"{key2separate}": 0, f"{key2color}": 0}
    uniq_values = {f"{key2separate}": 0, f"{key2color}": 0}
    has_nan[key2color], nan_inds[key2color], non_nan_values[key2color], uniq_values[
        key2color] = get_uniq_values_dealing_nan(
            data_dict, color_by=key2color
    )
    has_nan[key2separate], nan_inds[key2separate], non_nan_values[key2separate], uniq_values[
        key2separate] = get_uniq_values_dealing_nan(
            data_dict, color_by=key2separate
    )

    # Create a mapping from unique values to indices
    uniq2number = {}
    number2uniq = {}
    uniq2number[key2separate] = {str(value): idx for idx, value in
                                 enumerate(uniq_values[key2separate])}
    number2uniq[key2separate] = {val: str(key) for key, val in uniq2number[key2separate].items()}
    uniq2number[key2color] = {str(value): idx for idx, value in enumerate(uniq_values[key2color])}
    number2uniq[key2color] = {val: str(key) for key, val in uniq2number[key2color].items()}
    # Encode values
    reencode_values = np.array([uniq2number[key2color][str(val)] for val in data_dict[key2color]])
    # Assign colors to non-nan values and set the used color palette
    data_dict["colors_bk"], used_color_palette = assign_colors(
            uniq_values[key2color], reencode_values,
            if_color_gradient)
    if has_nan:  # set nan values to gray
        for ind in nan_inds[key2color]:
            data_dict["colors_bk"][ind] = "gray"

    cmap_interval = len(uniq_values[key2color]) - 1
    subfigs = []

    uniq_separate_keys = len(uniq2number[key2separate].keys())
    for jj, uniqname in enumerate(uniq2number[key2separate].keys()):
        ## plot the overall data in the first figure
        if jj == 0:
            # F1: scatter points figure + figure selection view + table + save button
            fig01, fig_select, table, savebutton = bokeh_plot_main_with_select_table(
                    data_dict,
                    cmap_interval=cmap_interval, number2uniq=number2uniq[key2color],
                    plot_height=plot_height, plot_width=plot_width, s2_hoverkeys=s2_hoverkeys,
                    scatter_size=scatter_size, table_columns=table_columns,
                    title=f"Over all color by {key2color} (n={len(data_dict['x'])})",
                    tooltips=tooltips, ticks_loc=np.linspace(
                            0, cmap_interval, len(uniq_values[key2color])),
                    used_color_palette=used_color_palette)
            subfigs += [fig01, fig_select, table, savebutton]

        ## plot subset of the data in the first figure
        current_inds = np.where(data_dict[key2separate] == uniqname)[0]
        sub_data_dict = update_sub_data_dict(
                data_dict, features[current_inds], needed_inds=current_inds,
                if_indiv_proj=if_indiv_proj)

        fig01_sub, fig_select_sub, table_sub, savebutton_sub = (
                bokeh_plot_main_with_select_table(
                        sub_data_dict,
                        cmap_interval=cmap_interval, number2uniq=number2uniq[key2color],
                        plot_height=plot_height, plot_width=plot_width, s2_hoverkeys=s2_hoverkeys,
                        scatter_size=scatter_size, table_columns=table_columns,
                        title=f"{uniqname} (n={len(sub_data_dict['x'])})",
                        tooltips=tooltips, ticks_loc=np.linspace(
                                0, cmap_interval, len(uniq_values[key2color])),
                        used_color_palette=used_color_palette))
        subfigs += [fig01_sub, fig_select_sub, table_sub, savebutton_sub]
    layout2 = grid(subfigs, ncols=4)
    output_file(
            save_dir + f'/{postfix}-sep_{uniq_separate_keys}-{key2separate}.html',
            title=title)
    bokeh_save(layout2)

    print("ok")


def update_sub_data_dict(data_dict, features, needed_inds, if_indiv_proj=False):

    sub_data_dict = {}
    for data_name in data_dict.keys():
        sub_data_dict[data_name] = np.array(data_dict[data_name])[needed_inds]
    if if_indiv_proj and features is not None:
        indiv_proj = get_reduced_dimension_projection(
                features, vis_mode="phate", n_components=2,
                n_neighbors=min(30, features.shape[0])
        )
        sub_data_dict["x"] = indiv_proj[:, 0]
        sub_data_dict["y"] = indiv_proj[:, 1]
    return sub_data_dict

"""Bokeh fuctions finished"""


def visualize_data_with_meta(features, meta_data_df, meta2check=[], cmap="viridis", n_cols=None,
                             n_rows=None,
                             postfix="STAD", figsize=[8, 6], save_dir="./", vis_mode="phate"):


    def is_df_column_string(df, column_name):
        col_datatype = df[column_name].dropna().dtypes
        return col_datatype == str or col_datatype == np.object_

    projection = get_reduced_dimension_projection(
        features, vis_mode=vis_mode,
        n_components=2,
        n_neighbors=30)

    if not n_cols:
        n_cols = np.int32(np.ceil(np.sqrt(len(meta2check))))
        n_rows = np.int32(np.ceil(len(meta2check) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    # Flatten the axes if it's not already a 1D array (in the case of a single plot)
    if n_rows * n_cols == 1:
        axes = [axes]  # Convert single axis to a list to be consistent with axes.flatten()
    else:
        axes = axes.flatten()
    plt.suptitle(postfix)
    for i, col in enumerate(meta2check):
        ax = axes[i]

        if is_df_column_string(meta_data_df, col):
            # Encode labels for string columns
            encoded_labels, classes = encode_labels(meta_data_df, col)
            color_labels = encoded_labels
            non_nan_classes = [ele for jj, ele in enumerate(classes) if str(ele) != "nan"]
        else:
            # Use the column values directly for numeric columns
            color_labels = meta_data_df[col].values
            non_nan_classes = np.unique(color_labels[~np.isnan(color_labels)])

        non_nan_inds = np.where(meta_data_df[col].values.astype(str) != "nan")[0]

        # Plot all points in gray
        ax.scatter(projection[:, 0], projection[:, 1], color="gray", alpha=0.65, s=5)

        # Plot non-NaN points with color
        sc = ax.scatter(
                projection[non_nan_inds, 0], projection[non_nan_inds, 1],
                c=color_labels[non_nan_inds], s=5, cmap=cmap)
        ax.set_title(textwrap.fill(str(col), width=30), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        # Add colorbar to the current axes
        cbar = fig.colorbar(sc, ax=ax)

        # Set the ticks and labels for the colorbar
        if is_df_column_string(meta_data_df, col):
            cbar.set_ticks(np.arange(len(non_nan_classes)))
            wrapped_labels = [textwrap.fill(str(label), width=25) for label in non_nan_classes]
            cbar.set_ticklabels(wrapped_labels, fontsize=8)

    # Adjust layout for better fit
    plt.tight_layout()
    plt.savefig(path.join(save_dir, f"vis_meta_{postfix}.png"))
    plt.close()

    return projection


def plot_individual_cancer_project(features_df,
                                   label_dict, projection, key2group="disease_code",
                                   if_indiv_proj=False,
                                   key2color="binary_response", prefix="prefix", save_dir="./"
                                   ):
    """
    plot each key2group in each subplot, use key2color to color code the points.
    :param label_dict:
    :param projection:
    :param key2group:
    :param key2color:
    :param prefix:
    :param save_dir:
    :return:
    """
    from sklearn.preprocessing import LabelEncoder
    uniq_disease = np.unique(label_dict[key2group])
    num_gene = features_df.shape[1]

    ## reencode str annotations to int label
    encoded_labels = {}
    if label_dict[key2color].dtype == 'object':
        # If the column has string values, encode them as numeric values
        le = LabelEncoder()
        reencode_color = le.fit_transform(label_dict[key2color])
        # Store the original and encoded values
        encoded_labels[key2color] = dict(zip(le.classes_, range(len(le.classes_))))
    else:
        reencode_color = np.array(label_dict[key2color]).astype(np.float32)
        uniq_lb = np.unique(label_dict[key2color])
        # Store the original and encoded values
        encoded_labels[key2color] = dict(zip(uniq_lb, range(len(uniq_lb))))

    # make figure
    row, col = find_number_of_subplots_closest_factors(len(uniq_disease) + 1)
    fig, axes = plt.subplots(row, col)
    if row * col == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()
    # Create scatter plot with all data and all labels for global colorbar
    scatter = axes_flat[0].scatter(
            projection[:, 0], projection[:, 1], c=reencode_color, cmap='viridis', s=5,
            label="all"
    )

    # Add colorbar
    cbar = fig.colorbar(
            scatter, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1, aspect=50,
    )
    cbar.set_ticks(np.arange(len(encoded_labels[key2color])))
    cbar.set_ticklabels(encoded_labels[key2color], rotation=45, ha="right")
    axes_flat[0].legend(frameon=True, fontsize=8)
    global_min = np.min(reencode_color)
    global_max = np.max(reencode_color)
    # Set original value labels on top of colorbar
    if col in encoded_labels:
        cbar.set_ticks(list(encoded_labels[col].values()))
        cbar.set_ticklabels(list(encoded_labels[col].keys()))
        cbar.ax.xaxis.set_ticks_position('top')  # Move the ticks to the top
    plt.suptitle(f"Original data proj. {prefix}")

    for jj, cancer in enumerate(uniq_disease):
        disease_inds = np.where(np.array(label_dict[key2group]) == cancer)[0]
        non_nan_inds = [ind for ind, ele in
                        enumerate(np.array(label_dict[key2group])[disease_inds]) if
                        str(ele) != "nan"]

        if not if_indiv_proj:
            plot_scatter_points_in_1_axe(
                    axes_flat[jj + 1], projection[disease_inds], non_nan_inds, prefix=cancer,
                    global_max=global_max, global_min=global_min,
                    colors=reencode_color[disease_inds])
        else:
            try:
                indiv_projection = get_reduced_dimension_projection(
                        features_df.values[:, -num_gene:][disease_inds], vis_mode="phate",
                        n_components=2,
                        n_neighbors=30
                )
                plot_scatter_points_in_1_axe(
                        axes_flat[jj + 1], indiv_projection, non_nan_inds, prefix=cancer,
                        global_max=global_max, global_min=global_min,
                        colors=reencode_color[disease_inds])
            except Exception as e:
                print(f"{cancer} has {len(disease_inds)} samples, No Phate possible! Error: {e}")
                plot_scatter_points_in_1_axe(
                        axes_flat[jj + 1], projection[disease_inds], non_nan_inds, prefix=cancer,
                        global_max=global_max, global_min=global_min,
                        colors=reencode_color[disease_inds])

    # Ensure that everything fits properly
    plt.tight_layout(rect=[0, 0.25, 1, 1])  # Leave space at the bottom for the colorbar
    plt.savefig(
            path.join(
                    save_dir,
                    f"{prefix}-indi-{key2group}-color{key2color}-indiProj"
                    f"{np.int32(if_indiv_proj)}.png"))

    plt.close()


def find_number_of_subplots_closest_factors(N):
    # Start with the square root of N
    sqrt_N = int(np.sqrt(N))

    # Initialize variables to store the closest factors
    row, col = sqrt_N, sqrt_N

    # Find the closest factors
    while row * col < N:
        if col <= row:
            col += 1
        else:
            row += 1

    return row, col


def plot_scatter_points_in_1_axe(axe, plot_projection, non_nan_inds, prefix="", global_max=None,
                                 global_min=None,
                                 colors=[0, 1, 1, 2, 0]):

    axe.scatter(
            plot_projection[:, 0], plot_projection[:, 1],
            c="gray", s=5, vmin=global_min, vmax=global_max
    )
    sc = axe.scatter(
            plot_projection[:, 0][non_nan_inds],
            plot_projection[:, 1][non_nan_inds],
            c=colors[non_nan_inds], vmin=global_min, vmax=global_max,
            label=f"{prefix}-{len(plot_projection)}", s=5
    )
    axe.legend(frameon=True, fontsize=8)

    axe.set_xticks([])
    axe.set_yticks([])