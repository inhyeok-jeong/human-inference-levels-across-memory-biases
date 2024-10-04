import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

from utils import *
from Model_Selection import *

#date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
date = '2023-11-23'
dir_base = f'./Model Selection/models'
dir_name = f"{dir_base}/epoch={epoch}"
if os.path.isdir(f"./Model Selection/output/{date}/epoch={epoch}"): # path duplication check
    k = 1
    while os.path.isdir(f"./Model Selection/output/{date}/epoch={epoch}" + f'-{k}'):
        k += 1
    os.mkdir(f"./Model Selection/output/{date}/epoch={epoch}" + f'-{k}')
else:
    os.mkdir(dir_base)

def load_model(survey_code, model_type):
    model_dict = {
        "NN": NN,
        "ratio-NN": ratio_NN,
        "Meta": [meta_LSTM, meta_NN],
        "ratio-Meta": [meta_ratio_LSTM, meta_ratio_NN],
        "KNN": KNN_noise
    }
    if model_type == "NN" or model_type == "ratio-NN":
        model = model_dict[model_type]()
        model.load_state_dict(torch.load(f"{dir_name}/{model_type}_{survey_code}" + ".pt"))
        model.eval()
    elif model_type == "Meta" or model_type == "ratio-Meta":
        model_LSTM = model_dict[model_type][0]()
        model_LSTM.load_state_dict(torch.load(f"{dir_name}/{model_type}_LSTM_{survey_code}" + ".pt"))
        model_LSTM.eval()
        
        model_NN = model_dict[model_type][1]()
        model_NN.load_state_dict(torch.load(f"{dir_name}/{model_type}_NN_{survey_code}" + ".pt"))
        model_NN.eval()
        model = [model_LSTM, model_NN]
    else:  
        model = model_dict[model_type]
    return model

def calc_reward_distribution(action_list):
    avg_value = sum(action_list) / len(action_list)
    dist = [abs(action - (2/3)*avg_value) for action in action_list]
    reward_distribution = [(1 if min(dist) == dist[idx] else 0) for idx in range(10)]
    return reward_distribution


def main_compete_NN(models, population, fig_title=None, action_limit=True):
    plt.clf()
    initial_actions = [random.randint(0, 100) for _ in range(10)]
    prev_actions = initial_actions
    actions_log = []
    rewards_log = []
    for contest_idx in range(5):
        for round_idx in range(10):
            actions = []
            for agent_idx in range(10):
                #action = models[agent_idx](torch.tensor(ordering_action_list(prev_actions, agent_idx))).item()
                # Abstract: API : input reference, output action through model.
                if type(models[agent_idx]) is list:
                    if population[agent_idx] == 'ratio-Meta':
                        x = models[agent_idx][0](torch.tensor(get_previous_behavior_sequence(actions_log, idx = round_idx, to_ratio = True))).detach()[0].numpy()
                        x = np.append(x, calc_inference_order(prev_actions[-1], prev_actions[round_idx - 1] if round_idx > 0 else ([100] * 10)))
                        action = models[agent_idx][1]().item()
                    else:
                        #print(models[agent_idx][0](torch.tensor(get_previous_behavior_sequence(prev_actions, idx = round_idx))).detach()[0])
                        x = models[agent_idx][0](torch.tensor(get_previous_behavior_sequence(actions_log, idx = round_idx))).detach()[0].numpy()
                        x = np.append(x, prev_actions[-1])
                        action = models[agent_idx][1](x).item()
                else:
                    if population[agent_idx] == 'ratio-NN':
                        action = models[agent_idx](torch.tensor(prev_actions + [calc_rewarded_action(prev_actions)])).item()
                    elif population[agent_idx] == 'NN':
                        action = models[agent_idx](torch.tensor(prev_actions + [calc_rewarded_action(prev_actions)])).item()
                    else:
                        action = KNN_noise(prev_actions, behavior_actions_dict[survey_code])
                if action_limit:
                    if action < 0:
                        action = 0
                    elif action > 100:
                        action = 100
                    else:
                        action = round(action, 2)
                actions.append(action)
            rewards = calc_reward_distribution(actions)
            actions_log.append(actions)
            rewards_log.append(rewards)
            prev_actions = actions
    
    #actions_log_array = np.array(actions_log)
    #rewards_log_array = np.array(rewards_log)

    return actions_log, rewards_log
#


if __name__ == "__main__":
    match_dict = read_match_dict("2023-11-12", epoch, 2)

    survey_codes = list(match_dict.keys())
    
    behavior_raw_data = read_behavior_data("./data", verbose=False)
    # for knn
    behavior_raw_actions = {}
    for survey_code in survey_codes:
        behavior_raw_actions[survey_code] = compose_behavior_data(behavior_raw_data[survey_code])
    behavior_actions_dict = {}
    for survey_code in survey_codes:
        behavior_actions_dict[survey_code] = compose_behavior_actions(behavior_raw_data[survey_code], behavior_raw_actions[survey_code])
    

    # Read Trained Model selected
    trained_model = {}
    for survey_code in survey_codes:
        trained_model[survey_code] = load_model(survey_code, match_dict[survey_code])
    
    f = open(f"./Model Selection/output/2023-11-23/epoch={epoch}/competition.txt", "w")

    log_list = []
    loop_num = 822
    for loop in range(loop_num):
        if loop % 50 == 0:
            print(f"[{loop}/{loop_num}]")
        population = random.sample(survey_codes, 10)
        sample_models = [trained_model[survey_code] for survey_code in population]
        
        actions_log_array, rewards_log_array = main_compete_NN(sample_models, population)

        f.write(f"{population}\n")
        f.write(f"{actions_log_array}\n")
        f.write(f"{rewards_log_array}\n")
        
    f.close()