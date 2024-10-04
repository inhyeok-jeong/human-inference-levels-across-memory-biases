import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from utils import *


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
    output_dir = "/".join([args.output_dir, "fig3", f"{args.font}_{args.font_size}"])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    match_dict = read_match_dict(dest_dir, epoch, 2)

    factors = read_factors()

    subject_data = read_competition_data(dest_dir, epoch)

    reward_case_cnt_dict = dict()
    for survey_code in subject_data.keys():
        reward_case_cnt_dict[survey_code] = []

    agent_ratio_list_dict = {
        "Meta": [],
        "ratio-NN": [],
        "KNN": []
    }

    for k in range(1, 50):
        E_p_value = []
        E_X_dict = dict()
        E_p_value_dict = dict()

        cnt_dict = {
            "ratio-NN": 0,
            "Meta": 0,
            "KNN": 0
        }

        for model_type in ['Meta', 'ratio-NN', 'KNN']:
            E_X_dict[model_type] = []
            E_p_value_dict[model_type] = []
        for survey_code in subject_data.keys():
            E_no_reward = []
            E_reward = []
            for sample_actions, sample_rewards, sample_avg in zip(subject_data[survey_code]['actions'], subject_data[survey_code]['rewards'], subject_data[survey_code]['avg']):        
                for idx in range(k - 1, len(sample_actions) - 1):
                    if sum(sample_rewards[idx - k + 1:idx + 1]) == 0:
                        #E_no_reward.append(sample_actions[idx + 1] / sample_actions[idx])
                        E_no_reward.append(sample_actions[idx + 1] / sample_avg)
                    else:
                        #E_reward.append(sample_actions[idx + 1] / sample_actions[idx])
                        E_reward.append(sample_actions[idx + 1] / sample_avg)

            #print(f"{survey_code}: {len(E_no_reward)}, {len(E_reward)}")
            reward_case_cnt_dict[survey_code].append(len(E_reward) / (len(E_reward) + len(E_no_reward)))
            pvalue = ttest_ind(E_no_reward, E_reward, equal_var=False).pvalue
            #print(pvalue)
            E_p_value.append(pvalue)

            # for coloring
            E_X_dict[match_dict[survey_code]].append(factors["y'_avg"][survey_code])
            E_p_value_dict[match_dict[survey_code]].append(pvalue)

            if pvalue < 0.05:
                cnt_dict[match_dict[survey_code]] += 1

        # print(f"[window size: {k}] {cnt_dict}")

        for model_type in ['Meta', 'ratio-NN', 'KNN']:
            agent_ratio_list_dict[model_type].append(cnt_dict[model_type]/len(E_X_dict[model_type]))
            

        #print(E_p_value)
        # print(min(E_p_value))
    plt.clf()
    #plt.scatter([factors["y'_avg"][survey_code] for survey_code in subject_data.keys()], E_p_value, s=dot_size)
    table = []
    for model_type in ['ratio-NN', 'KNN', 'Meta']:
        # original version
        # table.append(agent_ratio_list_dict[model_type])
        # delta version
        new_line = [0] + [(agent_ratio_list_dict[model_type][idx + 1] - agent_ratio_list_dict[model_type][idx]) for idx in range(len(agent_ratio_list_dict[model_type]) - 1)]
        table.append(new_line[:30]) # [:-1]

    table = np.array(table, dtype=np.float32)
    table = np.round(table, 2)
    
    fig, ax = plt.subplots(figsize=(20, 5))
    im = ax.imshow(table, cmap="coolwarm")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(table[0])), labels=[i for i in range(1, len(table[0]) + 1)])
    #ax.set_yticks(np.arange(6), labels=[4*i for i in range(6, 0, -1)])
    ax.set_yticks(np.arange(3), labels=convert_models2name(['ratio-NN', 'KNN', 'Meta']))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(3):
        for j in range(len(table[0])):
            text = ax.text(j, i, table[i, j],
                        ha="center", va="center", color="w")

    #ax.set_title("Agent ratio(p-value < 0.05)")
    ax.set_xlabel("Window size")
    ax.set_ylabel("Memory type")
    #fig.tight_layout()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Figure 3.png")
    plt.savefig(f"{output_dir}/Figure 3.pdf")

    # For checking the number of reward cases along with window size
    '''
    plt.clf()

    for label, Y in zip(convert_models2name(["Meta", "ratio-NN", "KNN"]), table):
        plt.plot(range(1, 50), Y, label=convert_model2name(label), color=convert_model2color(label))

    plt.savefig(f"./Model Selection/output/{dest_dir}/epoch={epoch}/E_total_plot_delta.png")
    plt.savefig(f"./Model Selection/output/{dest_dir}/epoch={epoch}/E_total_plot_delta.pdf")
    '''