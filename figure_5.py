import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, wasserstein_distance
from scipy.spatial.distance import jensenshannon
import seaborn as sns
from statannot import add_stat_annotation
import pandas as pd

from utils import *

dot_size = 3
palette = {
    "Episodic": "Red",
    "Working": "Green",
    "Habit": "Blue",
    "Low": "Red",
    "High": "Green"
}

def arrange_data_order(data: list, order: list[str]):
    new_data = []
    for target in order:
        for row in data:
            if row[1] == target:
                new_data.append(row)
                break
    for row in data:
        if row not in new_data:
            new_data.append(row)

    return new_data

def box_plot_with_scatter(df: pd.DataFrame, X: str, Y: str, filename: str, box_pairs = []):
    plt.clf()
    ax = sns.boxplot(x=X, y=Y, data=df, palette=palette)
    
    column_values = df[X].unique()
    for i, column_value in enumerate(column_values):
        data_box = df[df[X] == column_value][X]

        plt.scatter(np.random.normal(i, 0.04, size=len(data_box)), data_box, s=7, c='grey', alpha=0.7)
        
    add_stat_annotation(ax, data=df, x=X, y=Y, 
                        box_pairs=box_pairs,
                        test="t-test_welch", text_format="star", loc="inside", 
                        pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, ""]])

    plt.savefig(f"{output_dir}/{filename}.pdf")
    plt.savefig(f"{output_dir}/{filename}.png")
    return

def box_plot(df: pd.DataFrame, X: str, Y: str, filename: str, box_pairs = []):
    # plt.clf()
    ax = sns.boxplot(x=X, y=Y, data=df, palette=palette)
            
    add_stat_annotation(ax, data=df, x=X, y=Y, 
                        box_pairs=box_pairs,
                        test="t-test_welch", text_format="star", loc="inside", 
                        pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, ""]])

    plt.savefig(f"{output_dir}/{filename}.pdf")
    plt.savefig(f"{output_dir}/{filename}.png")
    return

def violin_plot_with_scatter(df: pd.DataFrame, X: str, Y: str, filename: str, box_pairs = []):
    # plt.clf()
    ax = sns.violinplot(x=X, y=Y, data=df, palette=palette, orient="v")
    
    column_values = df[X].unique()
    for i, column_value in enumerate(column_values):
        data_box = df[df[X] == column_value][X]

        plt.scatter(np.random.normal(i, 0.04, size=len(data_box)), data_box, s=7, c='grey', alpha=0.7)
        
    add_stat_annotation(ax, data=df, x=X, y=Y, 
                        box_pairs=box_pairs,
                        test="t-test_welch", text_format="star", loc="inside", 
                        pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, ""]])

    plt.savefig(f"{output_dir}/{filename}.pdf")
    plt.savefig(f"{output_dir}/{filename}.png")
    return

def violin_plot(df: pd.DataFrame, X: str, Y: str, filename: str, box_pairs = []):
    # plt.clf()
    ax = sns.violinplot(x=X, y=Y, data=df, palette=palette, orient="v")
            
    add_stat_annotation(ax, data=df, x=X, y=Y, 
                        box_pairs=box_pairs,
                        test="t-test_welch", text_format="star", loc="inside", 
                        pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, ""]])

    plt.savefig(f"{output_dir}/{filename}.pdf")
    plt.savefig(f"{output_dir}/{filename}.png")
    return

def box_plot_fig_5_a():
    # plt.clf()
    data = []
    for survey_code in subject_data.keys():
        A_1_1 = []
        for sample in subject_data[survey_code]['actions']:
            #print(sample)
            try:
                val = list(sample).index(0)
                A_1_1.append(list(sample).index(0))
            except:
                A_1_1.append(51)
                pass
            
        if len(A_1_1) != 0:
            data.append([survey_code, convert_model2name(match_dict[survey_code]), sum(A_1_1) / len(A_1_1)])

    df = pd.DataFrame(columns=["survey_code", "Memory type", "First zero round"], data=data)
    box_pairs = [("Episodic", "Working"), ("Working", "Habit"), ("Episodic", "Habit")]
    box_plot(df, "Memory type", "First zero round", "fig 5(a)", box_pairs=box_pairs)    

def box_plot_fig_5_a_violin():
    # plt.clf()
    data = []
    for survey_code in subject_data.keys():
        A_1_1 = []
        for sample in subject_data[survey_code]['actions']:
            #print(sample)
            try:
                val = list(sample).index(0)
                A_1_1.append(list(sample).index(0))
            except:
                A_1_1.append(51)
                pass
            
        if len(A_1_1) != 0:
            data.append([survey_code, convert_model2name(match_dict[survey_code]), sum(A_1_1) / len(A_1_1)])

    new_data = []
    for label in ["Working", "Habit", "Episodic"]:
        for row in data:
            if row[1] == label:
                new_data.append(row)
                break
    for row in data:
        if row not in new_data:
            new_data.append(row)
    data = new_data

    df = pd.DataFrame(columns=["survey_code", "Memory type", "First zero round"], data=data)
    box_pairs = [("Episodic", "Working"), ("Working", "Habit"), ("Episodic", "Habit")]
    violin_plot_with_scatter(df, "Memory type", "First zero round", "fig 5(a)", box_pairs=box_pairs)

def box_plot_fig_5_c():
    # plt.clf()
    data = []
    for survey_code in subject_data.keys():
        V_list = []
        for sample in subject_data[survey_code]['actions']:
            cnt = 0
            for idx in range(len(sample) - 1):
                if sample[idx] < sample[idx + 1]:
                    cnt += 1
            V_list.append(cnt)
        
        if len(V_list) != 0:
            data.append([survey_code, convert_model2name(match_dict[survey_code]), sum(V_list) / len(V_list)])

    df = pd.DataFrame(columns=["survey_code", "Memory type", "Number of value-increase rounds"], data=data)
    box_pairs = [("Episodic", "Working"), ("Working", "Habit"), ("Episodic", "Habit")]
    box_plot_with_scatter(df, "Memory type", "Number of value-increase rounds", "fig 5(c)", box_pairs=box_pairs)

def box_plot_fig_5_c_violin():
    # plt.clf()
    data = []
    for survey_code in subject_data.keys():
        V_list = []
        for sample in subject_data[survey_code]['actions']:
            cnt = 0
            for idx in range(len(sample) - 1):
                if sample[idx] < sample[idx + 1]:
                    cnt += 1
            V_list.append(cnt)
        
        if len(V_list) != 0:
            data.append([survey_code, convert_model2name(match_dict[survey_code]), sum(V_list) / len(V_list)])

    new_data = []
    for label in ["Working", "Habit", "Episodic"]:
        for row in data:
            if row[1] == label:
                new_data.append(row)
                break
    for row in data:
        if row not in new_data:
            new_data.append(row)
    data = new_data

    df = pd.DataFrame(columns=["survey_code", "Memory type", "Number of value-increase rounds"], data=data)

    box_pairs = [("Episodic", "Working"), ("Working", "Habit"), ("Episodic", "Habit")]
    violin_plot_with_scatter(df, "Memory type", "Number of value-increase rounds", "fig 5(c)", box_pairs=box_pairs)

def box_plot_fig_5_d():
    # plt.clf()
    threshold = sorted(factors["y'_avg"].values())[len(factors["y'_avg"].values()) // 2]
    data = []
    for survey_code in subject_data.keys():
        V_list = []
        for sample in subject_data[survey_code]['actions']:
            cnt = 0
            for idx in range(len(sample) - 1):
                if sample[idx] < sample[idx + 1]:
                    cnt += 1
            V_list.append(cnt)
        
        if len(V_list) != 0:
            data.append([survey_code, "high" if factors["y'_avg"][survey_code] >= threshold else "low", sum(V_list) / len(V_list)])

    df = pd.DataFrame(columns=["survey_code", "y'_avg", "Number of value-increase rounds"], data=data)
    box_pairs = [("low", "high")]
    box_plot_with_scatter(df, "y'_avg", "Number of value-increase rounds", "fig 5(d)", box_pairs=box_pairs)

def box_plot_fig_5_d_violin():
    # plt.clf()
    threshold = sorted(factors["y'_avg"].values())[len(factors["y'_avg"].values()) // 2]
    data = []
    for survey_code in subject_data.keys():
        V_list = []
        for sample in subject_data[survey_code]['actions']:
            cnt = 0
            for idx in range(len(sample) - 1):
                if sample[idx] < sample[idx + 1]:
                    cnt += 1
            V_list.append(cnt)
        
        if len(V_list) != 0:
            data.append([survey_code, "high" if factors["y'_avg"][survey_code] >= threshold else "low", sum(V_list) / len(V_list)])

    df = pd.DataFrame(columns=["survey_code", "y'_avg", "Number of value-increase rounds"], data=data)
    box_pairs = [("low", "high")]
    violin_plot_with_scatter(df, "y'_avg", "Number of value-increase rounds", "fig 5(d)", box_pairs=box_pairs)

def violin_plot_fig_5_noi_corr():
    # plt.clf()
    threshold = sorted(factors["y'_avg"].values())[len(factors["y'_avg"].values()) // 2]
    data = []
    for survey_code in subject_data.keys():
        V_list = []
        for sample in subject_data[survey_code]['actions']:
            cnt = 0
            for idx in range(len(sample) - 1):
                if sample[idx] < sample[idx + 1]:
                    cnt += 1
            V_list.append(cnt)
        
        if len(V_list) != 0:
            data.append([survey_code, convert_model2name(match_dict[survey_code])+"/"+("high" if factors["y'_avg"][survey_code] >= threshold else "low"), sum(V_list) / len(V_list)])

    data = arrange_data_order(data, ["Episodic/low", "Episodic/high", "Habit/low", "Habit/high", "Working/low", "Working/high"])

    df = pd.DataFrame(columns=["survey_code", "Memory type + y'_avg", "Number of value-increase rounds"], data=data)
    '''palette = {
        "low": "Red",
        "high": "Green"
    }'''
    box_pairs = []
    for y_1 in ["low", "high"]:
        for model_type_1 in ["Episodic", "Working", "Habit"]:
            for y_2 in ["low", "high"]:
                for model_type_2 in ["Episodic", "Working", "Habit"]:
                    if y_1 == y_2 and model_type_1 == model_type_2:
                        continue
                    if (model_type_2+"/"+y_2, model_type_1+"/"+y_1) in box_pairs:
                        continue
                    box_pairs.append((model_type_1+"/"+y_1, model_type_2+"/"+y_2))
    ax = sns.violinplot(x=f"Memory type + y'_avg", y=f"Number of value-increase rounds", data=df)
    '''for i in range(2):
        data_box = df[df["y'_avg"] == ["low", "high"][i]][["Number of value-increase rounds"]]

        plt.scatter(np.random.normal(i, 0.04, size=len(data_box)), data_box, s=7, c='grey', alpha=0.7)'''
    add_stat_annotation(ax, data=df, x=f"Memory type + y'_avg", y=f"Number of value-increase rounds", 
                        box_pairs=box_pairs,
                        test="t-test_welch", text_format="star", loc="inside", 
                        pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, ""]])

    plt.savefig(f"{output_dir}/fig 5 number of value-increase rounds corr.pdf")
    plt.savefig(f"{output_dir}/fig 5 number of value-increase rounds corr.png")

def violin_plot_fig_5_fzr_corr():
    # plt.clf()
    threshold = sorted(factors["y'_avg"].values())[len(factors["y'_avg"].values()) // 2]
    data = []
    for survey_code in subject_data.keys():
        A_1_1 = []
        for sample in subject_data[survey_code]['actions']:
            #print(sample)
            try:
                val = list(sample).index(0)
                A_1_1.append(list(sample).index(0))
            except:
                A_1_1.append(51)
                pass
            
        if len(A_1_1) != 0:
            data.append([survey_code, convert_model2name(match_dict[survey_code])+"/"+("high" if factors["y'_avg"][survey_code] >= threshold else "low"), sum(A_1_1) / len(A_1_1)])

    data = arrange_data_order(data, ["Episodic/low", "Episodic/high", "Habit/low", "Habit/high", "Working/low", "Working/high"])

    df = pd.DataFrame(columns=["survey_code", "Memory type + y'_avg", "First zero round"], data=data)
    box_pairs = []
    for y_1 in ["low", "high"]:
        for model_type_1 in ["Episodic", "Working", "Habit"]:
            for y_2 in ["low", "high"]:
                for model_type_2 in ["Episodic", "Working", "Habit"]:
                    if y_1 == y_2 and model_type_1 == model_type_2:
                        continue
                    if (model_type_2+"/"+y_2, model_type_1+"/"+y_1) in box_pairs:
                        continue
                    box_pairs.append((model_type_1+"/"+y_1, model_type_2+"/"+y_2))
    ax = sns.violinplot(x=f"Memory type + y'_avg", y=f"First zero round", data=df)
    '''for i in range(2):
        data_box = df[df["y'_avg"] == ["low", "high"][i]][["First zero round"]]

        plt.scatter(np.random.normal(i, 0.04, size=len(data_box)), data_box, s=7, c='grey', alpha=0.7)'''
    add_stat_annotation(ax, data=df, x=f"Memory type + y'_avg", y=f"First zero round", 
                        box_pairs=box_pairs,
                        test="t-test_welch", text_format="star", loc="inside", 
                        pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, ""]])

    plt.savefig(f"{output_dir}/fig 5 first zero round corr.pdf")
    plt.savefig(f"{output_dir}/fig 5 first zero round corr.png")

def violin_plot_fig_5_noi_corr_2():
    # plt.clf()
    threshold = sorted(factors["y'_avg"].values())[len(factors["y'_avg"].values()) // 2]
    data = []

    table = dict()

    table["Number of value-increase rounds"] = [[], [], [], [], []]
    convert = {
        "Episodic": 0,
        "Working": 1,
        "Habit": 2,
        "low": 3,
        "high": 4
    }

    for survey_code in subject_data.keys():
        V_list = []
        for sample in subject_data[survey_code]['actions']:
            cnt = 0
            for idx in range(len(sample) - 1):
                if sample[idx] < sample[idx + 1]:
                    cnt += 1
            V_list.append(cnt)
        
        if len(V_list) != 0:
            data.append([survey_code, convert_model2name(match_dict[survey_code]), sum(V_list) / len(V_list)])
            data.append([survey_code, "high" if factors["y'_avg"][survey_code] >= threshold else "low", sum(V_list) / len(V_list)])

            table["Number of value-increase rounds"][convert["high" if factors["y'_avg"][survey_code] >= threshold else "low"]].append(sum(V_list) / len(V_list))
            table["Number of value-increase rounds"][convert[convert_model2name(match_dict[survey_code])]].append(sum(V_list) / len(V_list))

    data = arrange_data_order(data, ["Episodic", "Habit", "Working", "low", "high"])

    df = pd.DataFrame(columns=["survey_code", "Memory type + y'_avg", "Number of value-increase rounds"], data=data)
    '''palette = {
        "low": "Red",
        "high": "Green"
    }'''
    box_pairs = []
    for y_1 in ["low", "high"]:
        for model_type_1 in ["Episodic", "Working", "Habit"]:
            box_pairs.append((model_type_1, y_1))
    ax = sns.violinplot(x=f"Memory type + y'_avg", y=f"Number of value-increase rounds", data=df)
    '''for i in range(2):
        data_box = df[df["y'_avg"] == ["low", "high"][i]][["Number of value-increase rounds"]]

        plt.scatter(np.random.normal(i, 0.04, size=len(data_box)), data_box, s=7, c='grey', alpha=0.7)'''
    add_stat_annotation(ax, data=df, x=f"Memory type + y'_avg", y=f"Number of value-increase rounds", 
                        box_pairs=box_pairs,
                        test="t-test_welch", text_format="star", loc="inside", 
                        pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, ""]])

    plt.savefig(f"{output_dir}/fig 5 number of value-increase rounds corr_2.pdf")
    plt.savefig(f"{output_dir}/fig 5 number of value-increase rounds corr_2.png")

    series = dict()
    for val_label in ["Number of value-increase rounds"]:
        series[val_label] = dict()
        for target in ["Episodic", "Working", "Habit", "low", "high"]:
           series[val_label][target] = pd.Series(table[val_label][convert[target]])

    groups = ["Episodic", "Working", "Habit", "low", "high"]

    for key in ["Number of value-increase rounds"]:
        print(f"{key} corr")
        for i in range(5):
            for j in range(i + 1, 5):
                print(f"{groups[i]}, {groups[j]} corr: {series[key][groups[i]].corr(series[key][groups[j]], method='pearson')}")
        print("================")

def violin_plot_fig_5_fzr_corr_2():
    plt.clf()
    threshold = sorted(factors["y'_avg"].values())[len(factors["y'_avg"].values()) // 2]
    data = []

    table = dict()

    table["First zero round"] = [[], [], [], [], []]
    convert = {
        "Episodic": 0,
        "Working": 1,
        "Habit": 2,
        "low": 3,
        "high": 4
    }

    for survey_code in subject_data.keys():
        A_1_1 = []
        for sample in subject_data[survey_code]['actions']:
            #print(sample)
            try:
                val = list(sample).index(0)
                A_1_1.append(list(sample).index(0))
            except:
                A_1_1.append(51)
                pass
            
        if len(A_1_1) != 0:
            data.append([survey_code, convert_model2name(match_dict[survey_code]), sum(A_1_1) / len(A_1_1)])
            data.append([survey_code, "high" if factors["y'_avg"][survey_code] >= threshold else "low", sum(A_1_1) / len(A_1_1)])

            table["First zero round"][convert["high" if factors["y'_avg"][survey_code] >= threshold else "low"]].append(sum(A_1_1) / len(A_1_1))
            table["First zero round"][convert[convert_model2name(match_dict[survey_code])]].append(sum(A_1_1) / len(A_1_1))

    data = arrange_data_order(data, ["Episodic", "Habit", "Working", "low", "high"])

    df = pd.DataFrame(columns=["survey_code", "Memory type + y'_avg", "First zero round"], data=data)
    '''palette = {
        "low": "Red",
        "high": "Green"
    }'''
    box_pairs = []
    for y_1 in ["low", "high"]:
        for model_type_1 in ["Episodic", "Working", "Habit"]:
            box_pairs.append((model_type_1, y_1))
    ax = sns.violinplot(x=f"Memory type + y'_avg", y=f"First zero round", data=df)
    '''for i in range(2):
        data_box = df[df["y'_avg"] == ["low", "high"][i]][["First zero round"]]

        plt.scatter(np.random.normal(i, 0.04, size=len(data_box)), data_box, s=7, c='grey', alpha=0.7)'''
    add_stat_annotation(ax, data=df, x=f"Memory type + y'_avg", y=f"First zero round", 
                        box_pairs=box_pairs,
                        test="t-test_welch", text_format="star", loc="inside", 
                        pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, ""]])

    plt.savefig(f"{output_dir}/fig 5 first zero round corr 2.pdf")
    plt.savefig(f"{output_dir}/fig 5 first zero round corr 2.png")

    series = dict()
    for val_label in ["First zero round"]:
        series[val_label] = dict()
        for target in ["Episodic", "Working", "Habit", "low", "high"]:
           series[val_label][target] = pd.Series(table[val_label][convert[target]])

    groups = ["Episodic", "Working", "Habit", "low", "high"]

    for key in ["First zero round"]:
        print(f"{key} corr")
        for i in range(5):
            for j in range(i + 1, 5):
                print(f"{groups[i]}, {groups[j]} corr: {series[key][groups[i]].corr(series[key][groups[j]], method='pearson')}")
        print("================")

def violin_plot_fig_5_mean():
    plt.clf()
    threshold = sorted(factors["y'_avg"].values())[len(factors["y'_avg"].values()) // 2]
    data = []

    table = dict()

    table["mean"] = [[], [], [], [], []]
    table["std 1"] = [[], [], [], [], []]
    table["std 2"] = [[], [], [], [], []]
    convert = {
        "Episodic": 0,
        "Working": 1,
        "Habit": 2,
        "low": 3,
        "high": 4
    }


    for survey_code in subject_data.keys():
        V_list = []
        V_raw = []
        for sample in subject_data[survey_code]['actions']:
            V_list.append(avg(sample))
            for v in sample:
                V_raw.append(v)

        if len(V_list) != 0:
            data.append([survey_code, convert_model2name(match_dict[survey_code]), sum(V_list) / len(V_list), np.std(np.array(V_list)), np.std(np.array(V_raw))])
            data.append([survey_code, "high" if factors["y'_avg"][survey_code] >= threshold else "low", sum(V_list) / len(V_list), np.std(np.array(V_list)), np.std(np.array(V_raw))])

            table["mean"][convert["high" if factors["y'_avg"][survey_code] >= threshold else "low"]].append(sum(V_list) / len(V_list))
            table["mean"][convert[convert_model2name(match_dict[survey_code])]].append(sum(V_list) / len(V_list))

            table["std 1"][convert["high" if factors["y'_avg"][survey_code] >= threshold else "low"]].append(np.std(np.array(V_list)))
            table["std 1"][convert[convert_model2name(match_dict[survey_code])]].append(np.std(np.array(V_list)))

            table["std 2"][convert["high" if factors["y'_avg"][survey_code] >= threshold else "low"]].append(np.std(np.array(V_raw)))
            table["std 2"][convert[convert_model2name(match_dict[survey_code])]].append(np.std(np.array(V_raw)))


    df = pd.DataFrame(columns=["survey_code", "Memory type + y'_avg", "mean", "std 1", "std 2"], data=data)
    box_pairs = []
    for y_1 in ["low", "high"]:
        for model_type_1 in ["Episodic", "Working", "Habit"]:
            box_pairs.append((model_type_1, y_1))

    violin_plot_with_scatter(df, "Memory type + y'_avg", "mean", "fig 5(mean mean)", box_pairs)
    violin_plot_with_scatter(df, "Memory type + y'_avg", "std 1", "fig 5(mean std 1)", box_pairs)
    violin_plot_with_scatter(df, "Memory type + y'_avg", "std 2", "fig 5(mean std 2)", box_pairs)

    series = dict()
    for val_label in ["mean", "std 1", "std 2"]:
        series[val_label] = dict()
        for target in ["Episodic", "Working", "Habit", "low", "high"]:
           series[val_label][target] = pd.Series(table[val_label][convert[target]])

    groups = ["Episodic", "Working", "Habit", "low", "high"]

    for key in ["mean", "std 1", "std 2"]:
        print(f"{key} corr")
        for i in range(5):
            for j in range(i + 1, 5):
                print(f"{groups[i]}, {groups[j]} corr: {series[key][groups[i]].corr(series[key][groups[j]], method='pearson')}")
        print("================")

def fig_5_distribution():
    plt.clf()
    threshold = sorted(factors["y'_avg"].values())[len(factors["y'_avg"].values()) // 2]
    data = []
    cnt = dict()
    for key_1 in ["Episodic", "Working", "Habit"]:
        for key_2 in ["low", "high"]:
            cnt[key_1+"/"+key_2] = 0

    for survey_code in subject_data.keys():
        A_1_1 = []
        for sample in subject_data[survey_code]['actions']:
            #print(sample)
            try:
                val = list(sample).index(0)
                A_1_1.append(list(sample).index(0))
            except:
                A_1_1.append(51)
                pass
            
        if len(A_1_1) != 0:
            data.append([survey_code, convert_model2name(match_dict[survey_code]), "High" if factors["y'_avg"][survey_code] >= threshold else "Low", sum(A_1_1) / len(A_1_1)])
            cnt[convert_model2name(match_dict[survey_code])+"/"+("high" if factors["y'_avg"][survey_code] >= threshold else "low")] += 1

    print(cnt)

def hist_fig_5_fzr(stat = "count", hue = "Memory type", ax=None):
    if ax is None:
        plt.clf()
        fig, ax = plt.subplots()
    threshold = sorted(factors["y'_avg"].values())[len(factors["y'_avg"].values()) // 2]
    data = []
    for survey_code in subject_data.keys():
        A_1_1 = []
        for sample in subject_data[survey_code]['actions']:
            #print(sample)
            try:
                val = list(sample).index(0)
                A_1_1.append(list(sample).index(0))
            except:
                A_1_1.append(51)
                pass
            
        if len(A_1_1) != 0:
            data.append([survey_code, convert_model2name(match_dict[survey_code]), "High" if factors["y'_avg"][survey_code] >= threshold else "Low", convert_model2name(match_dict[survey_code]) + ("High" if factors["y'_avg"][survey_code] >= threshold else "Low"), sum(A_1_1) / len(A_1_1)])
    
    data = arrange_data_order(data, ["Working", "Habit", "Episodic"])

    df = pd.DataFrame(data, columns=["survey_code", "Memory type", "y'_avg", "Memory type + y'_avg", "First zero round"])

    bins = np.histogram_bin_edges(df["First zero round"], bins="auto")

    sns.histplot(df, ax=ax, x="First zero round", hue=hue, multiple="dodge", bins=bins, stat=stat, common_norm=False, palette=palette)
    ax.set_ylabel("Proportion")
    # plt.savefig(f"{output_dir}/fig 5 hist fzr({stat}, {hue}).pdf")
    # plt.savefig(f"{output_dir}/fig 5 hist fzr({stat}, {hue}).png")

    probability_array = {}
    for pos, group in [("Memory type", "Episodic"), ("Memory type", "Habit"), ("Memory type", "Working"),
                       ("y'_avg", "High"), ("y'_avg", "Low")]:
        #v, _ = np.histogram(np.array(), bins=bins, density=True)
        v = [0] * 18
        for val in df[df[pos] == group]["First zero round"].tolist():
            v[round(val // 3)] += 1
        s = sum(v)
        v = [val / s for val in v]
        probability_array[group] = v

    return probability_array

def hist_fig_5_noi(stat = "count", hue = "Memory type", ax=None):
    if ax is None:
        plt.clf()
        fig, ax = plt.subplots()
    threshold = sorted(factors["y'_avg"].values())[len(factors["y'_avg"].values()) // 2]
    data = []
    for survey_code in subject_data.keys():
        V_list = []
        for sample in subject_data[survey_code]['actions']:
            cnt = 0
            for idx in range(len(sample) - 1):
                if sample[idx] < sample[idx + 1]:
                    cnt += 1
            V_list.append(cnt)
        
        if len(V_list) != 0:
            data.append([survey_code, convert_model2name(match_dict[survey_code]), "High" if factors["y'_avg"][survey_code] >= threshold else "Low", convert_model2name(match_dict[survey_code]) + ("High" if factors["y'_avg"][survey_code] >= threshold else "Low"), sum(V_list) / len(V_list)])
    
    data = arrange_data_order(data, ["Working", "Habit", "Episodic"])

    df = pd.DataFrame(data, columns=["survey_code", "Memory type", "y'_avg", "Memory type + y'_avg", "Number of value-increase rounds"])

    bins = np.histogram_bin_edges(df["Number of value-increase rounds"], bins="auto")
    sns.histplot(df, ax=ax, x="Number of value-increase rounds", hue=hue, bins=bins, multiple="dodge", stat=stat, common_norm=False, palette=palette)
    ax.set_ylabel("Proportion")
    # plt.savefig(f"{output_dir}/fig 5 hist noi({stat}, {hue}).pdf")
    # plt.savefig(f"{output_dir}/fig 5 hist noi({stat}, {hue}).png")
    
    probability_array = {}
    for pos, group in [("Memory type", "Episodic"), ("Memory type", "Habit"), ("Memory type", "Working"),
                       ("y'_avg", "High"), ("y'_avg", "Low")]:
        #v, _ = np.histogram(np.array(df[df[pos] == group]["Number of value-increase rounds"].tolist()), bins=bins, density=True)
        v = [0] * 18
        for val in df[df[pos] == group]["Number of value-increase rounds"].tolist():
            v[round(val // 3)] += 1
        s = sum(v)
        v = [val / s for val in v]
        probability_array[group] = v

    return probability_array

def likelihood_fig_5():
    plt.clf()
    threshold = sorted(factors["y'_avg"].values())[len(factors["y'_avg"].values()) // 2]
    data = []
    for survey_code in subject_data.keys():
        A_1_1 = []
        for sample in subject_data[survey_code]['actions']:
            #print(sample)
            try:
                val = list(sample).index(0)
                A_1_1.append(list(sample).index(0))
            except:
                A_1_1.append(51)
                pass
            
        if len(A_1_1) != 0:
            data.append([survey_code, convert_model2name(match_dict[survey_code]), sum(A_1_1) / len(A_1_1), "high" if factors["y'_avg"][survey_code] >= threshold else "low"])
    
    data = arrange_data_order(data, ["Episodic", "Habit", "Working"])

    df = pd.DataFrame(data, columns=["survey_code", "Memory type", "First zero round", "y'_avg"])

    likelihood = dict()
    for yp_avg in ["high", "low"]:
        for memory_type in ["Episodic", "Habit", "Working"]:
            condition_X = df["y'_avg"] == yp_avg
            condition_Y = df["Memory type"] == memory_type

            X_count = df[condition_X].shape[0]
            X_Y_count = df[condition_X & condition_Y].shape[0]

            likelihood[f"{memory_type} | {yp_avg}"] = X_Y_count / X_count
            
            condition_X = df["Memory type"] == memory_type
            condition_Y = df["y'_avg"] == yp_avg

            X_count = df[condition_X].shape[0]
            X_Y_count = df[condition_X & condition_Y].shape[0]

            likelihood[f"{yp_avg} | {memory_type}"] = X_Y_count / X_count


    # print(f"likelihood: {likelihood}")

    LL_diff = []
    for item in ["Working", "Habit", "Episodic"]:
        LL_diff.append(likelihood[f"high | {item}"] - likelihood[f"low | {item}"])

    fig, ax = plt.subplots()
    container = ax.bar(["Working", "Habit", "Episodic"], LL_diff, color = {"Working": "green", "Habit": "blue", "Episodic": "red"}.values())
    # ax.set_ylim(bottom = , top = )
    # ax.set_yticks([])
    plt.ylabel("Likelihood difference")
    plt.savefig(f"{output_dir}/likelihood diff bar.png")
    plt.savefig(f"{output_dir}/likelihood diff bar.pdf")

    #plt.savefig(f"./Model Selection/output/{dest_dir}/epoch={epoch}/fig 5 likelihood fzr.pdf")
    #plt.savefig(f"./Model Selection/output/{dest_dir}/epoch={epoch}/fig 5 likelihood fzr.png")
    return

def hist_js(array_dict, col, ax=None):
    # plt.cla()
    array = []
    # print("hist_js called")
    for key_1 in ["Working", "Habit", "Episodic"]:
        val = []
        for key_2 in ["Low", "High"]:
            val.append(jensenshannon(array_dict[key_1], array_dict[key_2]))
        array.append(val[0] - val[1])
    
    if ax is None:
        fig, ax = plt.subplots()
    container = ax.bar(["Working", "Habit", "Episodic"], array, color="grey")

    # plt.savefig(f"{output_dir}/histogram JS({col}).pdf")
    # plt.savefig(f"{output_dir}/histogram JS({col}).png")
    return

def hist_emd(array_dict, col):
    plt.cla()
    array = []
    # print("hist_emd called")
    for key_1 in ["Working", "Habit", "Episodic"]:
        val = []
        for key_2 in ["Low", "High"]:
            val.append(wasserstein_distance(array_dict[key_1], array_dict[key_2]))
        array.append(val[0] - val[1])
        
    fig, ax = plt.subplots()
    container = ax.bar(["Working", "Habit", "Episodic"], array, color = {"Working": "green", "Habit": "blue", "Episodic": "red"}.values())

    plt.savefig(f"{output_dir}/histogram EMD({col}).pdf")
    plt.savefig(f"{output_dir}/histogram EMD({col}).png")
    return

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
    output_dir = "/".join([args.output_dir, "fig5", f"{args.font}_{args.font_size}"])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    match_dict = read_match_dict(dest_dir, epoch, 2)

    factors = read_factors()

    subject_data = read_competition_data(dest_dir=dest_dir, epoch=epoch)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    #fig_5_distribution()
    for i, stat in enumerate(["probability"]): # ["count", "probability"]
        for j, group in enumerate(["Memory type"]): #, "y'_avg", "Memory type + y'_avg"]:
            r_1 = hist_fig_5_fzr(stat=stat, hue=group, ax=axes[0, 0])
            r_2 = hist_fig_5_noi(stat=stat, hue=group, ax=axes[1, 0])
        hist_js(r_1, col="fzr", ax=axes[0, 1])
        hist_js(r_2, col="noi", ax=axes[1, 1])
        plt.tight_layout()
        plt.savefig(f"{output_dir}/figure_5.png")
        plt.savefig(f"{output_dir}/figure_5.pdf")
    exit()
