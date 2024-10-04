import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.stats import ttest_ind
import seaborn as sns
from statannotations.Annotator import Annotator

from utils import *

def fig_4_a(ax=None, plot_scatter=False):
    if ax is None:
        plt.clf()

    df_data = []
    pivot_list = []
    for survey_code in match_dict.keys():
        if convert_model2name(match_dict[survey_code]) == "Working":
            df_data.append([convert_model2name(match_dict[survey_code]), survey_code, factors["y'_avg"][survey_code]])
            pivot_list.append(survey_code)
            break
    for survey_code in match_dict.keys():
        if convert_model2name(match_dict[survey_code]) == "Habit":
            df_data.append([convert_model2name(match_dict[survey_code]), survey_code, factors["y'_avg"][survey_code]])
            pivot_list.append(survey_code)
            break
    for survey_code in match_dict.keys():
        if convert_model2name(match_dict[survey_code]) == "Episodic":
            df_data.append([convert_model2name(match_dict[survey_code]), survey_code, factors["y'_avg"][survey_code]])
            pivot_list.append(survey_code)
            break
    for survey_code in match_dict.keys():
        if survey_code in pivot_list:
            continue
        df_data.append([convert_model2name(match_dict[survey_code]), survey_code, factors["y'_avg"][survey_code]])

    box_pairs = []
    for type_1 in convert_models2name(model_types):
        for type_2 in convert_models2name(model_types):
            if type_1 != type_2 and (type_2, type_1) not in box_pairs:
                box_pairs.append((type_1, type_2))
    box_pairs = [("Working", "Habit"), ("Working", "Episodic")]
    print(box_pairs)

    palette = dict()
    for type_ in model_types:
        palette[convert_model2name(type_)] = convert_model2color(type_)

    df = pd.DataFrame(columns=["Memory type", "survey_code", "y'_avg"], data=df_data)
    #ax = sns.boxplot(x="Memory type", y="y'_avg", data=df, palette=palette)#, order=["episodic", "working", "habitual"])
    ax = sns.violinplot(ax = ax, x=f"Memory type", y="y'_avg", data=df, palette=palette, orient="v")
    for i in range(len(model_types)):
        data_box = df[df["Memory type"] == convert_model2name(model_types[i])][["y'_avg"]]

        if plot_scatter:
            ax.scatter(np.random.normal(i, 0.04, size=len(data_box)), data_box, s=7, c='grey', alpha=0.7)
    annotator = Annotator(ax, box_pairs, data=df, x="Memory type", y="y'_avg")
    annotator.configure(test="t-test_welch", text_format="star", loc='inside', verbose=2)
    annotator.apply_and_annotate()
    '''add_stat_annotation(ax, data=df, x="Memory type", y="y'_avg",
                        box_pairs=box_pairs,
                        test="t-test_welch", text_format="star", loc="inside", 
                        pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, ""]])
    '''
    if ax is None:
        plt.ylabel("$y'_{avg}$")
        # plt.xlabel("Memory type")
    else:
        ax.set_ylabel("$y'_{avg}$")
        # ax.set_xlabel("Memory type")
        ax.set_xlabel("")
    # plt.savefig(f"{output_dir}/fig 4(a).png")
    # plt.savefig(f"{output_dir}/fig 4(a).pdf")

def fig_4_be(axes=None, plot_scatter=False):
    if axes is None:
        plt.clf()
    df_list = []
    for survey_code in sorted(match_dict.keys()):
        #df_list.append([match_dict[survey_code]+"/"+ ("high" if factors["y'_avg"][survey_code] >= 0 else "low"), factors["Lambda"][survey_code], factors["Rho"][survey_code], factors["Mu"][survey_code], factors["M-ratio"][survey_code]])
        df_list.append([convert_model2name(match_dict[survey_code]), factors["Lambda"][survey_code], factors["Rho"][survey_code], factors["Mu"][survey_code], factors["M-ratio"][survey_code]])

    df = pd.DataFrame(df_list, columns=["Memory type", "Lambda", "Rho", "Mu", "M-ratio"])

    box_pairs = []
    for type_1 in convert_models2name(model_types):
        for type_2 in convert_models2name(model_types):
            if type_1 == type_2 or (type_2, type_1) in box_pairs:
                continue
            box_pairs.append((type_1, type_2))

    palette = {
        "Episodic": "red",
        "Working": "green",
        "Habit": "blue"
    }

    for i, y_key in enumerate(["Lambda", "Rho", "Mu", "M-ratio"]):
        if y_key == "Lambda":
            df_new = df[-5 < df["Lambda"]]
            df_new = df_new[df_new["Lambda"] < 30]
        elif y_key == "Rho":
            df_new = df[df["Rho"] < 50]

        if axes is not None:
            ax = axes[1 + i]
        else:
            ax = None
    
        #ax = sns.boxplot(x="Memory type", y=y_key, data=df_new, order=["Working", "Habit", "Episodic"], palette=palette)
        ax = sns.violinplot(ax=ax, x=f"Memory type", y=y_key, data=df_new, order=["Working", "Habit", "Episodic"], palette=palette, orient="v")
        for i in range(len(model_types)):
            data_box = df_new[df["Memory type"] == convert_model2name(model_types[i])][[y_key]]

            if plot_scatter:
                ax.scatter(np.random.normal(i, 0.04, size=len(data_box)), data_box, s=7, c='grey', alpha=0.7)
            ax.set_xlabel("")
        '''annotator = Annotator(ax, box_pairs, data=df_new, x="Memory type", y=y_key)
        annotator.configure(test="t-test_welch", text_format="star", loc='inside', verbose=2)
        annotator.apply_and_annotate()'''
        '''add_stat_annotation(ax, data=df_new, x="Memory type", y=y_key,
                            box_pairs=box_pairs,
                            test="t-test_welch", text_format="star", loc="inside", 
                            pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, ""]])'''
        # plt.savefig(f"{output_dir}/fig 4_{y_key}.png")
        # plt.savefig(f"{output_dir}/fig 4_{y_key}.pdf")
        # plt.clf()

if __name__ == "__main__":
    # Set argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--font", default="Arial")
    parser.add_argument("--font_size", type=int, default=12)
    parser.add_argument("--output_dir", default="output")

    args = parser.parse_args()
    plt.rcParams['font.family'] = args.font
    plt.rcParams['font.size'] = args.font_size
    # plt.rcParams['text.usetex'] = True
    output_dir = "/".join([args.output_dir, "fig4", f"{args.font}_{args.font_size}"])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    epoch = 30
    dir_name = '2023-11-23'
    dest_dir = dir_name
    
    factors = read_factors()

    match_dict = read_match_dict(dir_name, epoch, 2)
    print(f"{2}: {set(list(match_dict.values()))}")

    model_types = ["ratio-NN", "KNN", "Meta"]

    matched_list = {}
    for type_ in model_types:
        matched_list[type_] = []
    
    for survey_code in sorted(match_dict.keys()):
        matched_list[match_dict[survey_code]].append(survey_code)

    fig, axes = plt.subplots(1, 5, figsize=(27, 5))

    fig_4_a(ax=axes[0])
    fig_4_be(axes=axes)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure_4.png")
    plt.savefig(f"{output_dir}/figure_4.pdf")