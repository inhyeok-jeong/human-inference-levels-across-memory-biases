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
from classes import *
from learning import *

#date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
date = '2023-11-12'
dir_base = f'./Model Selection/output/{date}'
if os.path.isdir(dir_base): # path duplication check
    k = 1
    while os.path.isdir(dir_base[:-1] + f'-{k}'):
        k += 1
    os.mkdir(dir_base[:-1] + f'-{k}')
else:
    os.mkdir(dir_base)

#
k_fold_num = 5

models = dict()

accepted_model = dict()
# accepted_model["survey_code"]: model

def match_model(survey_code, loss_dict, white_list = None):
    best = None
    loss_min = 987654321
    # decide appropriate model
    for model_type in loss_dict.keys():
        if white_list != None and model_type not in white_list:
            continue
        if loss_min > loss_dict[model_type]:
            loss_min = loss_dict[model_type]
            best = model_type

    if white_list == None:
        accepted_model[survey_code] = models[survey_code][best]
    return best

def main(epoch = 1, load_epoch = None):
    global behavior_raw_data, behavior_raw_actions, behavior_actions_dict, behavior_data, survey_codes, models, model_dict, accepted_model
    if os.path.isdir(f"{dir_base}/epoch={epoch}"):
        print(f"[Directory Duplication] epoch={epoch} already exists")
        print("Figures and output will be overwritten.")
    else:
        os.mkdir(f"{dir_base}/epoch={epoch}")
    dir_main = f"{dir_base}/epoch={epoch}"
    print()
    behavior_raw_data = read_behavior_data("./data", verbose=False)
    behavior_data = dict()

    survey_codes = list(behavior_raw_data.keys())

    # for knn
    behavior_raw_actions = {}
    for survey_code in survey_codes:
        behavior_raw_actions[survey_code] = compose_behavior_data(behavior_raw_data[survey_code])
    behavior_actions_dict = {}
    for survey_code in survey_codes:
        behavior_actions_dict[survey_code] = compose_behavior_actions(behavior_raw_data[survey_code], behavior_raw_actions[survey_code])

    match_dict = read_match_dict(date, epoch, 2)

    chk_name = {
        "NN": "NN",
        "ratio-NN": "ratio-NN",
        "Meta": "Meta_LSTM",
        "ratio-Meta": "ratio-Meta_LSTM",
        "KNN": "KNN",
    }


    # train model
    code_cnt = 1
    code_cnt_max = len(survey_codes)
    for survey_code in survey_codes:
        #print(f"./Model Selection/models/epoch={epoch}/{chk_name[match_dict[survey_code]]}_{survey_code}.pt")
        #print(os.path.isfile(f"./Model Selection/models/epoch={epoch}/{chk_name[match_dict[survey_code]]}_{survey_code}.pt"))
        if os.path.isfile(f"./Model Selection/models/epoch={epoch}/{chk_name[match_dict[survey_code]]}_{survey_code}.pt"):
            print(f"[{code_cnt}/{code_cnt_max}] {survey_code} already trained.")
            code_cnt += 1
            continue

        train_agent_selected_model(survey_code, epoch=epoch, load_epoch=0, model_type=match_dict[survey_code])

        print(f"[{code_cnt}/{code_cnt_max}] Trained [survey_code: {survey_code}, best model: {match_dict[survey_code]}]")
        code_cnt += 1
    


if __name__ == '__main__':
    main(30)