import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.manifold import TSNE


from utils import *

LABELS = ['Lambda', 'Rho', 'Mu', '$y_{avg}$', '$y_{median}$', "$y'_{avg}$", "$y'_{median}$", "$y'_1$", "$y'_2$", "$y'_3$", "$y'_4$", "$y'_5$", "Meta-d'", "M-ratio"]

corr_filename = './corr_221115(n=274).txt'

def plot_heatmap_corr(source_dir, output_dir, labels, fig=None, ax=None):
    elements = [['.' for j in range(14)] for i in range(14)]
    file = open(source_dir, "r")
    lines = file.readlines()
    file.close()
    for i in range(14):
        elements[i] = lines[1 + i][:-1].split('\t')[1:]
    
    for i in range(14):
        for j in range(14):
            elements[i][j] = elements[i][j] if elements[i][j] != '' else np.nan

    table = elements
    table = np.array(table, dtype=np.float32)
    table = np.round(table, 2)
    table = np.abs(table)
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(table, cmap='coolwarm')
    plt.colorbar(im)
    
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(14), labels=labels)
    ax.set_yticks(np.arange(14), labels=labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    fig.tight_layout()
    if ax is None:
        fig.savefig(f"{output_dir}/correlation coefficient heatmap.pdf")
        fig.savefig(f"{output_dir}/correlation coefficient heatmap.png")
    return

def plot_tSNE(data, perplexity, dot_size, output_dir, axes=None):
    param = ["Lambda", "y'_avg", "M-ratio"]

    X = np.concatenate((data[:, 0:3], data[:, 7:12], data[:, -1:]), axis = 1)

    tsne = TSNE(
        n_components=2,
        init="random",
        random_state=0,
        perplexity=perplexity,
        n_iter=300,
    )

    duun_index = []

    ax = None

    for target in range(3):
        if axes is not None:
            ax = axes[1, target]
        
        if target == 0:
            y = data[:, 0]
            # print("target is Lambda")
        elif target == 1:
            y = np.average(data[:, 7:12], axis=1)
            # print("target is y'_avg")
        elif target == 2:
            y = data[:, -1]
            # print("target is M-ratio")
        
        mid = np.median(y)
        red = y < mid
        green = y >= mid

        Y = tsne.fit_transform(X)
        Y = [[i for i in j] for j in Y]
        Y = np.array(Y)

        distance_list = []
        red_points = Y[red]
        green_points = Y[green]
        for red_point in red_points:
            for green_point in green_points:
                distance_list.append(np.linalg.norm(red_point - green_point))
        distance = np.average(distance_list)
        std = np.std(distance_list)

        within_distance_list = []
        for points in [red_points, green_points]:
            for idx, point_1 in enumerate(points):
                for idx2, point_2 in enumerate(points):
                    if idx >= idx2:
                        continue
                    within_distance_list.append(np.linalg.norm(point_1 - point_2))

        within_max_distance = np.max(within_distance_list)

        if ax is None:
            plt.clf()
            plt.scatter(Y[red, 0], Y[red, 1], c="r", label="low", s=dot_size)
            plt.scatter(Y[green, 0], Y[green, 1], c="g", label="high", s=dot_size)
            plt.legend()
            plt.savefig(f"{output_dir}/t-SNE {param[target]}.png")
            plt.savefig(f"{output_dir}/t-SNE {param[target]}.pdf")
        else:
            ax.scatter(Y[red, 0], Y[red, 1], c="r", label="low", s=dot_size)
            ax.scatter(Y[green, 0], Y[green, 1], c="g", label="high", s=dot_size)
            ax.legend()

        labels = []
        for idx in range(len(Y)):
            if red[idx]:
                labels.append("low")
            else:
                labels.append("high")

        duun_index.append(distance / within_max_distance)
        print(f"duun_index for {param[target]}: {duun_index[-1]}")
    
    if axes is None:
        plt.clf()
        plt.ylabel("Duun index")
        plt.bar(param, duun_index)
        plt.ylim((0.25, 0.35))
        plt.savefig(f"{output_dir}/t-SNE Duun Index.png")
        plt.savefig(f"{output_dir}/t-SNE Duun Index.pdf")
    else:
        ax = axes[0, 2]
        ax.set_ylabel("Duun index")
        ax.bar(param, duun_index)
        ax.set_xticks([0, 1, 2], ["Lambda", "$y'_{avg}$", "M-ratio"])
        ax.set_ylim((0.25, 0.35))
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

    output_dir = "/".join([args.output_dir, "fig1", f"{args.font}_{args.font_size}"])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    

    # Plot figure 1
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    plot_heatmap_corr(corr_filename, output_dir, LABELS, fig=fig, ax=axes[0, 0])
    
    with open('./TOTAL_FACTORS_221115(n=274).txt', 'r') as file:
        data = []
        for line in file.readlines():
            elements = list(map(float, line[:-1].split("\t")[1:]))
            if elements[0] == np.nan:
                continue
        # [survey code, lambda, rho, mu, y_avg, y_med, y'_avg, y'_med, y'_1, y'_2, y'_3, y'_4, y'_5, ]
            data.append(elements)
        data = np.array(data)

    plot_tSNE(data, 30, 40, output_dir, axes=axes)
    plt.savefig(f"{output_dir}/figure_1.png")
    plt.savefig(f"{output_dir}/figure_1.pdf")