import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
from pandas import DataFrame
from sklearn.manifold import TSNE
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from utils import *

POSTFIX = ["", "st", "nd", "rd", "th", "th"]

filename = './TOTAL_FACTORS_221115(n=274).txt'
bin_size = 3

# make target indicies list
factor_names = ['Lambda', 'Rho', 'Mu', "y_avg", "y_medain", "y'_avg", "y'_median", "y'_1", "y'_2", "y'_3", "y'_4", "y'_5", "log_avg", "log_median", "Meta-d'", "Mratio"]

index = {}
for i in range(len(factor_names)):
    index[factor_names[i]] = i

factor_names_box = ["level-1", "level-2", "level-3", "level-4", "level-5"]
for i in range(len(factor_names_box)):
    index[factor_names_box[i]] = i

def scatter_plot(filename, output_dir, axes=None, row_idx=2):
    # reading TOTAL_FACTORS data
    file = open(filename, 'r')
    data = []
    for line in file.readlines():
        elements = list(map(float, line[:-1].split("\t")[1:]))
        if elements[0] == np.nan:
            continue
	# [survey code, lambda, rho, mu, y_avg, y_med, y'_avg, y'_med, y'_1, y'_2, y'_3, y'_4, y'_5, ]
        data.append(elements)
    file.close()
    data = np.array(data)
    # print(len(data))

    X = np.concatenate((data[:, 0:3], data[:, 7:12], data[:, -1:]), axis = 1)
    factors_data = data

    # reading log_order data(level)
    path = '.'
    file_name = 'log_order_221115(n=274)'
    file = {}
    data = {}
    file['raw'] = open('{}/{}.txt'.format(path, file_name), 'r')
    data['raw'] = []
    #convert txt to 2d array
    
    for target in ['raw']:
        cast = int if target == 'bin' else float
        for line in file[target].readlines():
            val = line
            if val[-1] == '\n':
                val = val[:-1]
            val = val.split('\t')[1:]
            val = [item for item in val if item != '' and item != '/']

            list_val = list(map(cast, val))
            data[target].append(list_val)
            

        file[target].close()
        data[target] = np.array(data[target])
    
    level_data = data['raw']
    level_data = DataFrame(level_data)
    
    
    # drop all rows that have nan values
    #level_data = level_data.dropna(axis=0, how='any')
    #print(f"size: {len(X[0])}")
    
    #test data consistency
    '''
    for idx, (x, y) in enumerate(zip(X[:, 5], X[:, 6])):
            if x > 1 and y < 0:
                print(f"irregular: {x}, {y}, y'_avg: {factors_data[idx, 6]}")
    exit()
    '''
    Y_level = []
    Y_y_prime = []

    corr_coef_level_list = []
    corr_coef_y_prime_list = []

    for target_idx in range(1, 5):
        if axes is not None:
            ax = axes[row_idx, target_idx - 1]
        else:
            fig, ax = plt.subplot()
        #plt.title("paired scatter plot")
        
        scatter = ax.scatter(X[:, 2 + target_idx], X[:, 2 + target_idx + 1], c="blue", label=f"y'")
        x = X[:, 2 + target_idx]
        y = X[:, 2 + target_idx + 1]
        x = DataFrame(x)
        y = DataFrame(y)

        regr = linear_model.LinearRegression()
        regr.fit(x.values.reshape(-1, 1), y)
        y_plot = regr.predict(x)
        # print(f"y'_{target_idx} / y'_{target_idx + 1}")
        # print(f"r2 score: {r2_score(x, y)}")
        ax.plot(x, y_plot, color='blue', linewidth=3)
        
        # scatter level points
        level_now = DataFrame(level_data.loc[:, target_idx - 1:target_idx]).replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
        #print(f"level-{target_idx}")
        #print(level_now)
        # print(f"raw level-{target_idx}, {target_idx + 1} data points: {len(level_now)}")
        #print(level_now)
        level_x = level_now.loc[:, target_idx - 1]
        level_y = level_now.loc[:, target_idx]
        level_scatter = ax.scatter(level_x, level_y, c="red", label=f"raw level")

        regr_level = linear_model.LinearRegression()
        regr_level.fit(level_x.values.reshape(-1, 1), level_y)
        y_plot_level = regr_level.predict(level_x.values.reshape(-1, 1))
        # print(f"raw level_{target_idx} / raw level_{target_idx + 1}")
        # print(f"r2 score: {r2_score(level_x, level_y)}")
        ax.plot(level_x, y_plot_level, color='red', linewidth=3)
        
        ax.plot([-4.3, 2.8], [-4.3, 2.8], color='black', linewidth=2, linestyle="dotted")

        corr_coef_level = np.corrcoef(level_x, level_y)[0, 1]
        corr_coef_y_prime = np.corrcoef(X[:, 2 + target_idx], X[:, 2 + target_idx + 1])[0, 1]

        Y_level.append(corr_coef_level)
        Y_y_prime.append(corr_coef_y_prime)
        
        # print(f"level_{target_idx} range: {np.min(level_x)} ~ {np.max(level_x)}")
        # print(f"level_{target_idx + 1} range: {np.min(level_y)} ~ {np.max(level_y)}")
        # print(f"y'_{target_idx} range: {np.min(X[:, 2 + target_idx])} ~ {np.max(X[:, 2 + target_idx])}")
        # print(f"y'_{target_idx + 1} range: {np.min(X[:, 2 + target_idx + 1])} ~ {np.max(X[:, 2 + target_idx + 1])}")

        # print("============")

        # Print it to check pearson correlation coefficient.
        # print("total range corr coef")
        # print(f"corr coef raw level: {corr_coef_level}")
        # print(f"corr coef y_prime: {corr_coef_y_prime}\n")
        corr_coef_level_list.append(corr_coef_level)
        corr_coef_y_prime_list.append(corr_coef_y_prime)

        level_x_total = level_x.to_numpy()
        level_y_total = level_y.to_numpy()
        y_prime_x_total = X[:, 2 + target_idx]
        y_prime_y_total = X[:, 2 + target_idx + 1]
        level_x = []
        level_y = []
        y_prime_x = []
        y_prime_y = []

        for idx, (x, y) in enumerate(zip(level_x_total, level_y_total)):
            if 0 <= x and x <= 2.5 and 0 <= y and y <= 2.5:
                level_x.append(x)
                level_y.append(y)
        
        for idx, (x, y) in enumerate(zip(y_prime_x_total, y_prime_y_total)):
            if 0 <= x and x <= 2.5 and 0 <= y and y <= 2.5:
                y_prime_x.append(x)
                y_prime_y.append(y)

        #print(level_x[(0 <= level_x) & (level_x <= 2.5)])
        corr_coef_level_limit = np.corrcoef(level_x, level_y)[0, 1]
        corr_coef_y_prime_limit = np.corrcoef(y_prime_x, y_prime_y)[0, 1]
        # print("[0, 2.5] range corr coef")
        # print(f"corr coef raw level: {corr_coef_level_limit}")
        # print(f"corr coef y_prime: {corr_coef_y_prime_limit}\n")

        #plt.show()
        ax.set_xlabel(f"Value measured in {target_idx}{POSTFIX[target_idx]} contest")
        ax.set_ylabel(f"Value measured in {target_idx + 1}{POSTFIX[target_idx + 1]} contest")
        ax.set_xlim([-4.3, 2.81])
        ax.set_ylim([-4.3, 2.81])
        ax.legend(handles=[scatter, level_scatter], loc='lower right')
        # plt.savefig(f"{output_dir}/y_prime_paired_scatter_plot_"+str(target_idx)+"-"+str(target_idx + 1)+".pdf")
        # plt.savefig(f"{output_dir}/y_prime_paired_scatter_plot_"+str(target_idx)+"-"+str(target_idx + 1)+".png")
        # plt.clf()

    return Y_level, Y_y_prime, corr_coef_level_list, corr_coef_y_prime_list
    
def plot_pearson_corr(Y_level, Y_y_prime, corr_coef_level_list, corr_coef_y_prime_list, output_dir):
    plt.clf()
    # plt.title("pearson correlation coefficient over contest")
    plt.plot([1, 2, 3, 4], Y_level, color="red", label="raw level")
    plt.plot([1, 2, 3, 4], Y_y_prime, color="blue", label="y'")
    plt.legend()
    plt.xticks([1, 2, 3, 4])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Supple 3.png")
    plt.savefig(f"{output_dir}/Supple 3.pdf")

    plt.clf()
    fig, ax = plt.subplots()
    container = ax.bar(["raw level", "$y'_{avg}$"], [sum(corr_coef_level_list)/len(corr_coef_level_list), sum(corr_coef_y_prime_list)/len(corr_coef_y_prime_list)])
    ax.set_ylim(bottom = 0.7, top= 0.9)
    ax.set_yticks([0.7, 0.75, 0.8, 0.85, 0.9])
    plt.ylabel("pearson correlation coefficient")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Supple 3 bar.png")
    plt.savefig(f"{output_dir}/Supple 3 bar.pdf")
    plt.clf()

def read_file(path='.', file_name='TOTAL_FACTORS_221115(n=274)', bin_size=bin_size):
    file = {}
    data = {}
    file['raw'] = open('{}/{}.txt'.format(path, file_name), 'r')
    file['bin'] = open('{}/{}_bin{}.txt'.format(path, file_name, bin_size), 'r')
    
    data['raw'] = []
    data['bin'] = []

    #convert txt to 2d array
    for target in ['raw', 'bin']:
        cast = int if target == 'bin' else float
        for line in file[target].readlines()[:]: ## custom n
            val = line.split('\t')[1:]
            if val[-1] == '\n':
                val = val[:-1]

            if val[-1][-1] == '\n':
                val[-1] = val[-1][:-1]

            data[target].append(list(map(cast, val)))
            

        file[target].close()
        data[target] = np.array(data[target])
        
    return data

# X, Y: str(factor name in factor_names)
def get_list_for_box(data, X, Y):
    X_val = data['bin'][:, index[X]]
    Y_val = data['raw'][:, index[Y]]

    result = [[] for i in range(bin_size)]

    # make data as 2d list
    for idx in range(len(X_val)):
        result[X_val[idx]-1].append(Y_val[idx])

    # print(min(result[2]), max(result[2]))
    # print(min(result[1]), max(result[1]))
    # print(min(result[0]), max(result[0]))

    return result
    
def draw_box_plot(data, X, Y, X_label, Y_label, output_dir, ax=None):
    # plt.clf()
    #plt.title('bin{} {}_distribution of {}'.format(bin_size, X, Y))
    data_for_df = []

    Xs = ["Low", "Mid", "High"]

    for idx, row in enumerate(data):
        #print(f"for loop : {Xs[idx]}")
        for v in row:
            if v == 20 or v == -20:
                continue
            data_for_df.append([Xs[idx], v])
    df = pd.DataFrame(columns=[f"{X}", f"{Y}"], data=data_for_df)
    # print(df)
    palette = dict()
    for i in range(bin_size):
        palette[i + 1] = "white"
    palette["Low"] = "white"
    palette["Mid"] = "white"
    palette["High"] = "white"

    box_pairs = []
    for x in range(1, bin_size + 1):
        if df[df[f"{X}"] == Xs[x - 1]].empty:
            continue
        for y in range(1, bin_size + 1):
            if df[df[f"{X}"] == Xs[y - 1]].empty:
                continue
            if x >= y:
                continue
            box_pairs.append((Xs[x - 1], Xs[y - 1]))

    # print(box_pairs)
    ax = sns.boxplot(ax=ax, x=f"{X}", y=f"{Y}", data=df, palette=palette)
    # plt.xlabel(X_label)
    # plt.ylabel(Y_label)
    #ax = sns.violinplot(x=f"{X}", y=f"{Y}", data=df, palette=palette)
    '''
    add_stat_annotation(ax, data=df, x=f"{X}", y=f"{Y}", 
                        box_pairs=box_pairs,
                        test="t-test_welch", text_format="star", loc="inside", 
                        pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, ""]])
    '''

    for i in range(bin_size):
        #data_box = [val for val in data[i] if -5 < val < 12]
        data_box = data[i]
        ax.scatter(np.random.normal(i, 0.04, size=len(data_box)), data_box, s=10, c='grey', alpha=1)
    #plt.show()
    if "level" not in X:
        ax.set_yticks([-3, -2, -1, 0, 1, 2])
        ax.set_ylim([-3.1, 2])
        #s = f"$y'_{int(X[-1])}$"
        #ax.set_xlabel(s)
        s = f"$y'_{int(Y[-1])}$"
        ax.set_ylabel(s)
        # plt.ylim([-5.1, 3])
    else:
        ax.set_yticks([-5, 0, 5, 10, 15])
        ax.set_ylim([-5, 15])
        # plt.xlabel(f"$y'_{int(X[-1])}$")
    ax.set_xlabel("")

    # To calculate overlap-related quantitative values
    iqr_low = df[df[X] == "Low"][Y].quantile(0.75) - df[df[X] == "Low"][Y].quantile(0.25)
    iqr_high = df[df[X] == "High"][Y].quantile(0.75) - df[df[X] == "High"][Y].quantile(0.25)

    low_upper_error_range = (round(df[df[X] == "Low"][Y].quantile(0.75), 3), round(df[df[X] == "Low"][Y].quantile(0.75) + 1.5 * iqr_low, 3))
    high_lower_error_range = (round(df[df[X] == "High"][Y].quantile(0.25) - 1.5 * iqr_high, 3), round(df[df[X] == "High"][Y].quantile(0.25), 3))

    print(f"X: {X}, Y: {Y}")
    if low_upper_error_range[1] < high_lower_error_range[0]:
        print(f"(1) No overlap | {low_upper_error_range}, {high_lower_error_range}")
    else:
        print(f"(1) overlap range: ({round(max(low_upper_error_range[0], high_lower_error_range[0]), 3)}, {round(min(low_upper_error_range[1], high_lower_error_range[1]), 3)})")
        print(f"    overlap length: {round(min(low_upper_error_range[1], high_lower_error_range[1]) - max(low_upper_error_range[0], high_lower_error_range[0]), 3)}")

    cnt_high_in_low_error = df[(df[X] == "High") & (df[Y] >= low_upper_error_range[0]) & (df[Y] <= low_upper_error_range[1])].shape[0]
    cnt_low_in_high_error = df[(df[X] == "Low") & (df[Y] >= high_lower_error_range[0]) & (df[Y] <= high_lower_error_range[1])].shape[0]
    cnt_high = df[df[X] == "High"].shape[0]
    cnt_low = df[df[X] == "Low"].shape[0]

    # print(f"(2) ratio of high bin dots in low's error bar: {round(cnt_high_in_low_error / cnt_high, 3)}")
    # print(f"(2) ratio of low bin dots in high's error bar: {round(cnt_low_in_high_error / cnt_low, 3)}")
    print(f"(2) ratio of overlapped dots: {round((cnt_high_in_low_error + cnt_low_in_high_error) / (cnt_low + cnt_high), 3)}")

    # plt.savefig('{}/bin{}_{}_{}.pdf'.format(output_dir, bin_size, X, Y))
    # plt.savefig('{}/bin{}_{}_{}.png'.format(output_dir, bin_size, X, Y))
    # plt.clf()

def plot_box_consistency(filename, params, labels, output_dir, axes=None, row_idx=0):
    p, q = filename.split("/")
    data = read_file(p, q, bin_size)

    for i, X in enumerate(params):
        for j, Y in enumerate(params):
            if X == Y:
                continue
            if int(X[-1]) + 1 != int(Y[-1]):
                continue
            X_label = labels[i]
            Y_label = labels[j]
            if axes is not None:
                draw_box_plot(get_list_for_box(data, X, Y), X, Y, X_label, Y_label, output_dir, ax=axes[row_idx, i])
            else:
                draw_box_plot(get_list_for_box(data, X, Y), X, Y, X_label, Y_label, output_dir)

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

    output_dir = "/".join([args.output_dir, "fig2", f"{args.font}_{args.font_size}"])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    

    # Plot figure 2
    fig, axes = plt.subplots(3, 4, figsize=(27, 15))
    plot_box_consistency('./TOTAL_FACTORS_221115(n=274)', ["y'_1", "y'_2", "y'_3", "y'_4", "y'_5"], ["$y'_1$", "$y'_2$", "$y'_3$", "$y'_4$", "$y'_5$"], output_dir, axes=axes, row_idx=0)
    plot_box_consistency('./log_order_221115(n=274)', ["level-1", "level-2", "level-3", "level-4", "level-5"], ["level-1", "level-2", "level-3", "level-4", "level-5"], output_dir, axes=axes, row_idx=1)
    Y_level, Y_y_prime, corr_coef_level_list, corr_coef_y_prime_list = scatter_plot(filename, output_dir, axes=axes, row_idx=2)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure_2.png")
    plt.savefig(f"{output_dir}/figure_2.pdf")

    plt.cla()
    fig, axes = plt.subplots()
    plot_pearson_corr(Y_level, Y_y_prime, corr_coef_level_list, corr_coef_y_prime_list, output_dir)