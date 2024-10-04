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
from learning import *

date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
dir_base = f'./Model Selection/output/{date}'
if os.path.isdir(dir_base): # path duplication check
    k = 1
    while os.path.isdir(dir_base[:-1] + f'-{k}'):
        k += 1
    os.mkdir(dir_base[:-1] + f'-{k}')
else:
    os.mkdir(dir_base)

def compose_training_data_k_fold(individual_behavior_data, k=5, batch_size = 1):
    train_loaders = []
    test_loaders = []
    for idx in range(k):
        training_X = []
        training_Y = []
        test_X = []
        test_Y = []
        for contest_idx in range(5):
            if contest_idx == idx:
                target_X = test_X
                target_Y = test_Y
            else:
                target_X = training_X
                target_Y = training_Y

            for round_idx in range(9):
                target_X.append(individual_behavior_data[contest_idx][round_idx])
                target_Y.append([individual_behavior_data[contest_idx][round_idx + 1][-1]])

        trainset = BehaviorDataset(torch.from_numpy(np.array(training_X)).float(),
                                    torch.from_numpy(np.array(training_Y)).float())
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = False)
        train_loaders.append(train_loader)
        
        testset = BehaviorDataset(torch.from_numpy(np.array(test_X)).float(),
                                    torch.from_numpy(np.array(test_Y)).float())
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle = False)
        test_loaders.append(test_loader)
    
    return train_loaders, test_loaders

def compose_training_data_ratio_k_fold(individual_behavior_data, k=5, batch_size = 1):
    train_loaders = []
    test_loaders = []
    for idx in range(k):
        training_X = []
        training_Y = []
        test_X = []
        test_Y = []
        for contest_idx in range(5):
            if contest_idx == idx:
                target_X = test_X
                target_Y = test_Y
            else:
                target_X = training_X
                target_Y = training_Y
            for round_idx in range(9):
                target_X.append(individual_behavior_data[contest_idx][round_idx] + [calc_rewarded_action(individual_behavior_data[contest_idx][round_idx])])
                if individual_behavior_data[contest_idx][round_idx + 1][-1] == 0:
                    target_Y.append([10]) 
                else:
                    val = ( np.log(individual_behavior_data[contest_idx][round_idx + 1][-1] / avg(individual_behavior_data[contest_idx][round_idx])) / 
                            np.log(calc_p(individual_behavior_data[contest_idx][round_idx], calc_rewarded_action(individual_behavior_data[contest_idx][round_idx]))))
                    if val < -10:
                        target_Y.append([-10])
                    elif val < 10:
                        target_Y.append([val])    
                    else:
                        target_Y.append([10])

        trainset = BehaviorDataset(torch.from_numpy(np.array(training_X)).float(),
                                    torch.from_numpy(np.array(training_Y)).float())
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = False)
        train_loaders.append(train_loader)
        
        testset = BehaviorDataset(torch.from_numpy(np.array(test_X)).float(),
                                    torch.from_numpy(np.array(test_Y)).float())
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle = False)
        test_loaders.append(test_loader)
    
    return train_loaders, test_loaders

def compose_training_data_LSTM_k_fold(individual_behavior_data, k=5, batch_size = 1, to_ratio = False):
    train_loaders = []
    test_loaders = []
    for idx in range(k):
        training_X = []
        training_Y = []
        test_X = []
        test_Y = []
        for contest_idx in range(5):
            if contest_idx == idx:
                target_X = test_X
                target_Y = test_Y
            else:
                target_X = training_X
                target_Y = training_Y

            for round_idx in range(9):
                target_X.append(get_previous_behavior_sequence(individual_behavior_data[contest_idx], round_idx + 1, 4, to_ratio=to_ratio))
                #target_X.append(individual_behavior_data[contest_idx][round_idx][:-1])
                if to_ratio:
                    target_Y.append([calc_inference_order(item, individual_behavior_data[contest_idx][round_idx]) for item in individual_behavior_data[contest_idx][round_idx + 1][:-1]])
                else:
                    target_Y.append([individual_behavior_data[contest_idx][round_idx + 1][:-1]])

        trainset = BehaviorDataset(torch.from_numpy(np.array(training_X)).float(),
                                    torch.from_numpy(np.array(training_Y)).float())
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = False)
        train_loaders.append(train_loader)
        
        testset = BehaviorDataset(torch.from_numpy(np.array(test_X)).float(),
                                    torch.from_numpy(np.array(test_Y)).float())
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle = False)
        test_loaders.append(test_loader)
    
    return train_loaders, test_loaders

def compose_training_data_LSTM_NN_k_fold(individual_behavior_data, LSTM_models = None, k=5, batch_size = 1):
    train_loaders = []
    test_loaders = []
    for idx in range(k):
        training_X = []
        training_Y = []
        test_X = []
        test_Y = []
        for contest_idx in range(5):
            if contest_idx == idx:
                target_X = test_X
                target_Y = test_Y
            else:
                target_X = training_X
                target_Y = training_Y

            for round_idx in range(9):
                if LSTM_models == None:
                    target_X.append(individual_behavior_data[contest_idx][round_idx + 1])
                else:
                    LSTM_model = LSTM_models[idx]
                    target_X.append(LSTM_model.forward(individual_behavior_data[contest_idx][round_idx]))
                target_Y.append([individual_behavior_data[contest_idx][round_idx + 1][-1]])

        trainset = BehaviorDataset(torch.from_numpy(np.array(training_X)).float(),
                                    torch.from_numpy(np.array(training_Y)).float())
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = False)
        train_loaders.append(train_loader)
        
        testset = BehaviorDataset(torch.from_numpy(np.array(test_X)).float(),
                                    torch.from_numpy(np.array(test_Y)).float())
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle = False)
        test_loaders.append(test_loader)
    
    return train_loaders, test_loaders

def compose_training_data_LSTM_NN_k_fold_each(individual_behavior_data, LSTM_model = None, idx=0, batch_size = 1, to_ratio = False):
    training_X = []
    training_Y = []
    test_X = []
    test_Y = []
    for contest_idx in range(5):
        if contest_idx == idx:
            target_X = test_X
            target_Y = test_Y
        else:
            target_X = training_X
            target_Y = training_Y

        for round_idx in range(9):
            if LSTM_model == None:
                target_X.append(individual_behavior_data[contest_idx][round_idx + 1][:-1])
            else:
                X = LSTM_model.forward(torch.tensor([get_previous_behavior_sequence(individual_behavior_data[contest_idx], round_idx + 1, 4, to_ratio=to_ratio)]))
                #X = LSTM_model.forward(torch.tensor([individual_behavior_data[contest_idx][round_idx][:-1]]))
                X = X.detach()[0][-1].numpy()
                #print("Target point")
                #print(X)
                if to_ratio:
                    ## TODO: is 100 valid / appropriate?
                    X = np.append(X, calc_inference_order(individual_behavior_data[contest_idx][round_idx][-1], individual_behavior_data[contest_idx][round_idx - 1] if round_idx > 0 else ([100] * 10))) # NN structure: LSTM prediction(1x9) + last action(1x1)
                else:
                    X = np.append(X, individual_behavior_data[contest_idx][round_idx][-1]) # NN structure: LSTM prediction(1x9) + last action(1x1)
                #print(X)
                target_X.append(X)
            target_Y.append([individual_behavior_data[contest_idx][round_idx + 1][-1]])

    trainset = BehaviorDataset(torch.from_numpy(np.array(training_X)).float(),
                                torch.from_numpy(np.array(training_Y)).float())
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = False)
    
    testset = BehaviorDataset(torch.from_numpy(np.array(test_X)).float(),
                                torch.from_numpy(np.array(test_Y)).float())
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle = False)
    
    return train_loader, test_loader

def compose_behavior_actions(individual_raw_data, individual_behavior_data):
    behavior_actions = []
    for contest_idx in range(5):

        if contest_idx > 0:
            actions = [individual_raw_data[11 * (contest_idx - 1) + agent_idx][-1] for agent_idx in range(10)] # magic number 10 is num of agents
            prev_avg = sum(actions) / len(actions)
            behavior_actions.append((prev_avg, individual_behavior_data[contest_idx][0][-1]))
            #print(f"{contest_idx}, 0 : {actions}, {prev_avg}, {individual_behavior_data[contest_idx][round_idx][-1]}")
        for round_idx in range(1, 10):
            actions = [individual_raw_data[11 * contest_idx + agent_idx][round_idx - 1] for agent_idx in range(10)] # magic number 10 is num of agents
            prev_avg = sum(actions) / len(actions)
            behavior_actions.append((prev_avg, individual_behavior_data[contest_idx][round_idx][-1]))
            #print(f"{contest_idx}, {round_idx} : {actions}, {prev_avg}, {individual_behavior_data[contest_idx][round_idx][-1]}")
    return behavior_actions


#
k_fold_num = 5

models = dict()
model_dict = {
    "NN": dict(),
    "ratio-NN": dict(),
    "Meta": dict(),
    "ratio-Meta": dict(),
    "KNN": dict(),
}
# models["survey_code"]["type"]: model_dict["type"]

accepted_model = dict()
# accepted_model["survey_code"]: model

def test_knn(survey_code, verbose = False):
    loss_list = []
    
    behavior_data_target = behavior_data[survey_code]
    for contest_idx in range(5):
        behavior_actions = filter_behavior_actions(behavior_actions_dict[survey_code], contest_idx)
        for round_idx in range(9):
            action = KNN_noise(behavior_data_target[contest_idx][round_idx], behavior_actions)
            loss_list.append(abs(action - behavior_data_target[contest_idx][round_idx + 1][-1]))

    loss_average = sum(loss_list) / len(loss_list)
    return loss_average

model_train_func = {
    "NN": train_NN,
    "ratio-NN": train_ratio_NN,
    "Meta": train_Meta,
    "ratio-Meta": train_ratio_Meta,
    "KNN": test_knn,
}

def train_agent(survey_code, epoch, load_epoch, verbose = False):
    k = 5

    global behavior_raw_data, behavior_data, behavior_raw_actions, behavior_actions_dict
    loss_dict = dict()
    for model_type in model_dict.keys():
        loss_dict[model_type] = []
    models[survey_code] = init_models()
    behavior_data[survey_code] = compose_behavior_data(behavior_raw_data[survey_code])

    train_loaders, test_loaders = compose_training_data_k_fold(behavior_data[survey_code])
    train_loaders_ratio, test_loaders_ratio = compose_training_data_ratio_k_fold(behavior_data[survey_code])
    train_loaders_LSTM, test_loaders_LSTM = compose_training_data_LSTM_k_fold(behavior_data[survey_code])
    train_loaders_ratio_LSTM, test_loaders_ratio_LSTM = compose_training_data_LSTM_k_fold(behavior_data[survey_code], to_ratio = True)

    if not os.path.isdir(f"./Model Selection/models/epoch={epoch+load_epoch}"):
        os.mkdir(f"./Model Selection/models/epoch={epoch+load_epoch}")

    for model_type in model_dict.keys():
        if model_type == "NN":
            if verbose:
                print(f"[{survey_code}] NN training")
            for train_loader, test_loader in zip(train_loaders, test_loaders):
                model = NN()
                #if load_epoch > 0:
                    #model.load_state_dict(torch.load(f"./Model Selection/models/epoch={load_epoch}/NN_{survey_code}.pt"))
                for _ in range(epoch):
                    loss_dict[model_type].append(model_train_func[model_type](survey_code, model, train_loader, test_loader, optim.Adam(model.parameters(), lr=0.001), epoch=epoch, verbose = False))
            torch.save(model.state_dict(), f"./Model Selection/models/epoch={epoch+load_epoch}/NN_{survey_code}.pt")
        elif model_type == "ratio-NN":
            if verbose:
                print(f"[{survey_code}] ratio-NN training")
            for train_loader, test_loader in zip(train_loaders_ratio, test_loaders_ratio):
                model = ratio_NN()
                for _ in range(epoch):
                    loss_dict[model_type].append(model_train_func[model_type](survey_code, model, train_loader, test_loader, optim.Adam(model.parameters(), lr=0.001), epoch=epoch, verbose = False))
            torch.save(model.state_dict(), f"./Model Selection/models/epoch={epoch+load_epoch}/ratio-NN_{survey_code}.pt")
        elif model_type == "Meta":
            if verbose:
                print(f"[{survey_code}] Meta training")
            i = 0
            for train_loader_LSTM, test_loader_LSTM in zip(train_loaders_LSTM, test_loaders_LSTM):
                model_LSTM = meta_LSTM()
                model_NN = meta_NN()
                for _ in range(epoch):
                    loss_dict[model_type].append(model_train_func[model_type](survey_code, model_LSTM, train_loader_LSTM, test_loader_LSTM, optim.Adam(model_LSTM.parameters(), lr=0.001), model_NN, optim.Adam(model_NN.parameters(), lr=0.001), idx=i, epoch=epoch, verbose = False))
                i += 1
            torch.save(model.state_dict(), f"./Model Selection/models/epoch={epoch+load_epoch}/Meta_{survey_code}.pt")
        elif model_type == "ratio-Meta":
            if verbose:
                print(f"[{survey_code}] ratio-Meta training")
            i = 0
            for train_loader_ratio_LSTM, test_loader_ratio_LSTM in zip(train_loaders_ratio_LSTM, test_loaders_ratio_LSTM):
                model_LSTM = meta_ratio_LSTM()
                model_NN = meta_ratio_NN()
                for _ in range(epoch):
                    loss_dict[model_type].append(model_train_func[model_type](survey_code, model_LSTM, train_loader_ratio_LSTM, test_loader_ratio_LSTM, optim.Adam(model_LSTM.parameters(), lr=0.001), model_NN, optim.Adam(model_NN.parameters(), lr=0.001), idx=i, epoch=epoch, verbose = False))
                i += 1
            torch.save(model.state_dict(), f"./Model Selection/models/epoch={epoch+load_epoch}/ratio-Meta_{survey_code}.pt")
        elif model_type == "KNN":
            if verbose:
                print(f"[{survey_code}] KNN testing")
            loss_dict[model_type].append(test_knn(survey_code))
            if verbose:
                print()
    
    loss_result = {}
    for model_type in model_dict.keys():
        if model_type == "KNN":
            loss_result[model_type] = loss_dict[model_type][0]
            continue
        L = len(loss_dict[model_type])
        loss_result[model_type] = sum([loss_dict[model_type][L // k * (i + 1) - 1] for i in range(k)]) / k
    #print(f"[{survey_code}] train complete.")
    return loss_result

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
    factors = read_factors()

    survey_codes = list(behavior_raw_data.keys())

    # for knn
    behavior_raw_actions = {}
    for survey_code in survey_codes:
        behavior_raw_actions[survey_code] = compose_behavior_data(behavior_raw_data[survey_code])
    behavior_actions_dict = {}
    for survey_code in survey_codes:
        behavior_actions_dict[survey_code] = compose_behavior_actions(behavior_raw_data[survey_code], behavior_raw_actions[survey_code])

    match_dict = dict()

    match_dicts = dict()
    white_lists = []
    types = ["NN", "ratio-NN", "Meta", "ratio-Meta", "KNN"]
    for i in range(5):
        for j in range(i + 1, 5):
            e = []
            for k in range(5):
                if k == i or k == j:
                    continue
                e.append(types[k])
            white_lists.append(e)

    for i in range(5):
        e = []
        for k in range(5):
            if k == i:
                continue
            e.append(types[k])
        white_lists.append(e)

    dict_idx = 0
    for white_list in white_lists:
        match_dicts[dict_idx] = dict()
        dict_idx += 1

    # train models using k-fold cross validation for model selection
    code_cnt = 1
    code_cnt_max = len(survey_codes)
    for survey_code in survey_codes:
        loss_dict = train_agent(survey_code, epoch=epoch, load_epoch=0)
        best_type = match_model(survey_code, loss_dict)
        match_dict[survey_code] = best_type
        dict_idx = 0
        for white_list in white_lists:
            match_dicts[dict_idx][survey_code] = match_model(survey_code, loss_dict, white_list = white_list)
            dict_idx += 1

        print(f"[{code_cnt}/{code_cnt_max}] survey_code: {survey_code}, best model: {best_type}, loss: {loss_dict[best_type]}")
        code_cnt += 1
    
    # plot figure
    Y = [random.random() for _ in survey_codes]
    plt.figure(figsize=(20, 10))
    plt.title("model matching result")
    plt.xlabel("y'_avg")
    codes_model = {}
    Y_model = {}
    for model_type in types:
        codes_model[model_type] = []
        Y_model[model_type] = []
    for i, survey_code in enumerate(survey_codes):
        codes_model[match_dict[survey_code]].append(survey_code)
        Y_model[match_dict[survey_code]].append(Y[i])

    for model_type in types:
        plt.scatter([factors["y'_avg"][survey_code] for survey_code in codes_model[model_type]], Y_model[model_type], c = [convert_type2color(match_dict[survey_code]) for survey_code in codes_model[model_type]], label = model_type)
    plt.legend(loc = 'lower left')
    # save figure
    plt.savefig(f"{dir_main}/model_matching_result.png")
    plt.close()

    with open(f"{dir_main}/value_log.txt", "w") as file:
        file.write(str(Y))
        file.write('\n')
        file.write(str([factors["y'_avg"][survey_code] for survey_code in survey_codes]))
        file.write('\n')
        file.write(str(match_dict))
        file.write('\n')
    
    for dict_idx in range(len(white_lists)):
        codes_model = {}
        Y_model = {}
        white_list = white_lists[dict_idx]
        for model_type in white_list:
            codes_model[model_type] = []
            Y_model[model_type] = []
        for i, survey_code in enumerate(survey_codes):
            codes_model[match_dicts[dict_idx][survey_code]].append(survey_code)
            Y_model[match_dicts[dict_idx][survey_code]].append(Y[i])

        plt.cla()
        plt.figure(figsize=(20, 10))
        plt.title("model matching result")
        plt.xlabel("y'_avg")
        for model_type in white_list:
            plt.scatter([factors["y'_avg"][survey_code] for survey_code in codes_model[model_type]], Y_model[model_type], c = [convert_type2color(match_dicts[dict_idx][survey_code]) for survey_code in codes_model[model_type]], label = model_type)
        plt.legend(loc = 'lower left')
        # save figure
        plt.savefig(f"{dir_main}/model_matching_result_{dict_idx}.png")
        plt.close()

    with open(f"{dir_main}/value_log.txt", "a") as file:
        file.write(str(match_dicts))
        file.write('\n')
            
    matched_list = {}
    for model_type in types:
        matched_list[model_type] = []
        for survey_code in survey_codes:
            if match_dict[survey_code] == model_type:
                matched_list[model_type].append(survey_code)
    '''
    # plot figure error bar version
    plt.cla()
    plt.figure(figsize=(20, 10))
    plt.title("model matching result")
    plt.ylabel("y'_avg")
    for i, model_type in enumerate(types):
        plt.boxplot([factors["y'_avg"][survey_code] for survey_code in matched_list[model_type]], positions=[i], widths=0.5, labels=[model_type])

    # p-test
    

    #plt.legend()
    # save figure
    plt.savefig(f"{dir_main}/model_matching_result_box.png")
    plt.close()
    
    for dict_idx in range(len(white_lists)):
        
        matched_list = {}
        white_list = white_lists[dict_idx]
        for model_type in white_list:
            matched_list[model_type] = []
            for survey_code in survey_codes:
                if match_dicts[dict_idx][survey_code] == model_type:
                    matched_list[model_type].append(survey_code)

        plt.cla()
        plt.figure(figsize=(20, 10))
        plt.title("model matching result")
        plt.ylabel("y'_avg")
        for i, model_type in enumerate(white_list):
            plt.boxplot([factors["y'_avg"][survey_code] for survey_code in matched_list[model_type]], positions=[i], widths=0.5, labels=[model_type])
        #plt.legend()
        # save figure
        plt.savefig(f"{dir_main}/model_matching_result_box_{dict_idx}.png")
        plt.close()
    '''

if __name__ == '__main__':
    main(30)