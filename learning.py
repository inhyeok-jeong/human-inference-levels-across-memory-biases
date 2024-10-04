import random

import numpy as np

from utils import *
from classes import *

def calc_rewarded_action(action_list):
    try:
        action_list = action_list.cpu().detach().numpy()
        #print(action_list)
    except:
        pass
       
    avg_value = sum(action_list) / len(action_list)
    dist = [(abs(action - (2/3)*avg_value), idx) for (idx, action) in enumerate(action_list)]
    return action_list[min(dist)[1]]

# return behavior_actions after except specific contest(test_contest_idx will be used for testing)
def filter_behavior_actions(behavior_actions, test_contest_idx):
    filtered_behavior_actions = []
    for contest_idx in range(5):
        if contest_idx == test_contest_idx:
            continue
        filtered_behavior_actions.append(behavior_actions[contest_idx])
    return filtered_behavior_actions

def KNN(last_actions, behavior_actions, k=3):
    #KNN: average of last round 
    avg = sum(last_actions) / len(last_actions)

    dist_action_list = []
    for behavior_action in behavior_actions:
        dist = abs(avg - behavior_action[0])
        dist_action_list.append((dist, behavior_action[1]))
    
    dist_action_list.sort(key=lambda x: x[0])
    dist_action_list = dist_action_list[:k]
    actions = [item[1] for item in dist_action_list]
    #print(f"{last_actions}, {behavior_actions}, {actions}")
    return sum(actions) / len(actions)

def KNN_noise(last_actions, behavior_actions, k=3):
    #KNN: average of last round 
    avg = sum(last_actions) / len(last_actions)

    dist_action_list = []
    for behavior_action in behavior_actions:
        dist = abs(avg - behavior_action[0])
        dist_action_list.append((dist, behavior_action[1]))
    
    dist_action_list.sort(key=lambda x: x[0])
    dist_action_all = dist_action_list
    dist_action_list = dist_action_list[:k]
    idx = 0
    while dist_action_list[-1][0] == dist_action_all[k + idx][0]:
        dist_action_list.append(dist_action_all[k + idx])
        idx += 1
    actions = [item[1] for item in dist_action_list]
    std = np.std(actions)
    #return median(actions) + random.normalvariate(0, std/2)
    return (sum(actions) / len(actions)) + random.normalvariate(0, std/2)

def calc_inference_order(action, last_actions):
    max_val = 10
    min_val = -10

    order = ( np.log(action / avg(last_actions)) / 
              np.log(calc_p(last_actions, calc_rewarded_action(last_actions))))
    
    if order < min_val:
        order = min_val
    elif order > max_val:
        order = max_val
    
    if action == 0 and avg(last_actions) == 0:
        ### MAYBE PROBLEMATIC
        order = max_val
    elif action == 0:
        order = max_val
    elif avg(last_actions) == 0:
        order = min_val
    
    return order

def get_previous_behavior_sequence(individual_behavior_data_contest, idx = 0, length = 4, to_ratio = False, debug = False):
    if debug:
        print(f"idx: {idx}, length: {length}")
        print(f"individual_behavior_data_contest: {individual_behavior_data_contest}")
    prev_sequence = []
    if idx < length:
        for _ in range(length - idx):
            if to_ratio:
                prev_sequence.append([0] * 9) # 9 is number of other agents
            else:
                prev_sequence.append([100] * 9) # 9 is number of other agents
    for round_idx in range(idx - length + len(prev_sequence), idx):
        element = individual_behavior_data_contest[round_idx][:-1]
        if to_ratio:
            # TODO : is 100 valid / appropriate?
            element = [calc_inference_order(item, (individual_behavior_data_contest[round_idx - 1] if round_idx > 0 else ([100] * 9))) for item in individual_behavior_data_contest[round_idx][:-1]]
        prev_sequence.append(element)
    
    return prev_sequence

def compose_training_data(individual_behavior_data, batch_size = 1):
    train_X = []
    train_Y = []
    for contest_idx in range(5):
        for round_idx in range(9):
            train_X.append(individual_behavior_data[contest_idx][round_idx])
            train_Y.append([individual_behavior_data[contest_idx][round_idx + 1][-1]])

    trainset = BehaviorDataset(torch.from_numpy(np.array(train_X)).float(),
                                torch.from_numpy(np.array(train_Y)).float())
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = False)

    return train_loader

def compose_training_data_ratio(individual_behavior_data, batch_size = 1):
    train_X = []
    train_Y = []
    for contest_idx in range(5):
        for round_idx in range(9):
            train_X.append(individual_behavior_data[contest_idx][round_idx] + [calc_rewarded_action(individual_behavior_data[contest_idx][round_idx])])
            if individual_behavior_data[contest_idx][round_idx + 1][-1] == 0:
                train_Y.append([10]) 
            else:
                val = ( np.log(individual_behavior_data[contest_idx][round_idx + 1][-1] / avg(individual_behavior_data[contest_idx][round_idx])) / 
                        np.log(calc_p(individual_behavior_data[contest_idx][round_idx], calc_rewarded_action(individual_behavior_data[contest_idx][round_idx]))))
                if val < -10:
                    train_Y.append([-10])
                elif val < 10:
                    train_Y.append([val])    
                else:
                    train_Y.append([10])

        trainset = BehaviorDataset(torch.from_numpy(np.array(train_X)).float(),
                                    torch.from_numpy(np.array(train_Y)).float())
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = False)

    return train_loader

def compose_training_data_LSTM(individual_behavior_data, batch_size = 1, to_ratio = False):
    train_X = []
    train_Y = []
    for contest_idx in range(5):
        for round_idx in range(9):
            train_X.append(get_previous_behavior_sequence(individual_behavior_data[contest_idx], round_idx + 1, 4, to_ratio=to_ratio))
            #train_X.append(individual_behavior_data[contest_idx][round_idx][:-1])
            if to_ratio:
                train_Y.append([calc_inference_order(item, individual_behavior_data[contest_idx][round_idx]) for item in individual_behavior_data[contest_idx][round_idx + 1][:-1]])
            else:
                train_Y.append([individual_behavior_data[contest_idx][round_idx + 1][:-1]])

        trainset = BehaviorDataset(torch.from_numpy(np.array(train_X)).float(),
                                    torch.from_numpy(np.array(train_Y)).float())
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = False)
     
    return train_loader

# Maybe deprecated.
def compose_training_data_LSTM_NN(individual_behavior_data, LSTM_model = None, k=5, batch_size = 1):
    train_X = []
    train_Y = []
    for contest_idx in range(5):
        for round_idx in range(9):
            if LSTM_model == None:
                train_X.append(individual_behavior_data[contest_idx][round_idx + 1])
            else:
                LSTM_model = LSTM_model
                train_X.append(LSTM_model.forward(individual_behavior_data[contest_idx][round_idx]))
            train_Y.append([individual_behavior_data[contest_idx][round_idx + 1][-1]])

    trainset = BehaviorDataset(torch.from_numpy(np.array(train_X)).float(),
                                torch.from_numpy(np.array(train_Y)).float())
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = False)
    
    return train_loader

def compose_training_data_LSTM_NN_each(individual_behavior_data, LSTM_model = None, idx=0, batch_size = 1, to_ratio = False):
    train_X = []
    train_Y = []
    for contest_idx in range(5):
        for round_idx in range(9):
            if LSTM_model == None:
                train_X.append(individual_behavior_data[contest_idx][round_idx + 1][:-1])
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
                train_X.append(X)
            train_Y.append([individual_behavior_data[contest_idx][round_idx + 1][-1]])

    trainset = BehaviorDataset(torch.from_numpy(np.array(train_X)).float(),
                                torch.from_numpy(np.array(train_Y)).float())
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = False)
    
    return train_loader

def init_NN():
    return NN()

def init_ratio_NN():
    return ratio_NN()

def init_Meta():
    return {
        "LSTM": meta_LSTM(),
        "NN": meta_NN()
    }

def init_ratio_Meta():
    return {
        "LSTM": meta_ratio_LSTM(),
        "NN": meta_ratio_NN()
    }

def init_KNN():
    return None

init_model_func = {
    "NN": init_NN,
    "ratio-NN": init_ratio_NN,
    "Meta": init_Meta,
    "ratio-Meta": init_ratio_Meta,
    "KNN": init_KNN,
}

def init_models():
    models = dict()
    for model_type in ["NN", "ratio-NN", "Meta", "ratio-Meta", "KNN"]:
        models[model_type] = init_model_func[model_type]()
    return models

def train_model(survey_code, model, train_loader, test_loader, optimzer, is_ratio = False, epoch = 1, verbose = False):
    model.train()
    device = 'cpu'
    loss_list = []

    for _ in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimzer.zero_grad()
            output = model(data)
            loss = F.l1_loss(output, target)
            #print(f"loss from {output} and {target}: {loss}")
            if type(loss) == torch.nan:  
                print("Error found")
                return
            loss.backward()
            optimzer.step()

    model.eval()
    return

train_NN = train_model

def train_ratio_NN(survey_code, model, train_loader, test_loader, optimzer, epoch=1, verbose = False):
    return train_model(survey_code, model, train_loader, test_loader, optimzer, is_ratio = True, epoch = epoch, verbose = verbose)

def train_Meta(survey_code, model_LSTM, train_loader_LSTM, test_loader_LSTM, optimizer_LSTM, model_NN, optimzer_NN, idx=0, epoch=1, verbose = False):
    loss_average = train_model(survey_code, model_LSTM, train_loader_LSTM, test_loader_LSTM, optimizer_LSTM, verbose = verbose)
    if verbose:
        print("LSTM train done")
    test_loader_LSTM_NN = None
    train_loader_LSTM_NN = compose_training_data_LSTM_NN_each(compose_behavior_data(behavior_raw_data[survey_code]), LSTM_model = model_LSTM)
    loss_average = train_model(survey_code, model_NN, train_loader_LSTM_NN, test_loader_LSTM_NN, optimzer_NN, epoch = epoch, verbose = verbose)
    return loss_average

def train_ratio_Meta(survey_code, model_LSTM, train_loader_LSTM, test_loader_LSTM, optimizer_LSTM, model_NN, optimzer_NN, idx=0, epoch=1, verbose = False):
    loss_average = train_model(survey_code, model_LSTM, train_loader_LSTM, test_loader_LSTM, optimizer_LSTM, is_ratio = True, verbose = verbose)
    if verbose:
        #print(loss_average)
        print("ratio-LSTM train done")
    test_loader_LSTM_NN = None
    train_loader_LSTM_NN = compose_training_data_LSTM_NN_each(compose_behavior_data(behavior_raw_data[survey_code]), LSTM_model = model_LSTM, to_ratio = True)
    loss_average = train_model(survey_code, model_NN, train_loader_LSTM_NN, test_loader_LSTM_NN, optimzer_NN, epoch = epoch, is_ratio = True, verbose = verbose)
    return loss_average

model_train_func = {
    "NN": train_NN,
    "ratio-NN": train_ratio_NN,
    "Meta": train_Meta,
    "ratio-Meta": train_ratio_Meta,
}

def train_agent_selected_model(survey_code, epoch, load_epoch, model_type, verbose = False):
    global behavior_raw_data, behavior_data, behavior_raw_actions, behavior_actions_dict
    loss_dict = dict()
    loss_dict[model_type] = []
    behavior_data[survey_code] = compose_behavior_data(behavior_raw_data[survey_code])

    train_loader = compose_training_data(behavior_data[survey_code])
    train_loader_ratio = compose_training_data_ratio(behavior_data[survey_code])
    train_loader_LSTM = compose_training_data_LSTM(behavior_data[survey_code])
    train_loader_ratio_LSTM = compose_training_data_LSTM(behavior_data[survey_code], to_ratio = True)
    test_loader = None
    test_loader_LSTM = None
    test_loader_ratio_LSTM = None

    if not os.path.isdir(f"./Model Selection/models/epoch={epoch+load_epoch}"):
        os.mkdir(f"./Model Selection/models/epoch={epoch+load_epoch}")

    if model_type == "NN":
        if verbose:
            print(f"[{survey_code}] NN training")
        model = NN()
        #if load_epoch > 0:
            #model.load_state_dict(torch.load(f"./Model Selection/models/epoch={load_epoch}/NN_{survey_code}.pt"))
        for _ in range(epoch):
            loss_dict[model_type].append(model_train_func[model_type](survey_code, model, train_loader, test_loader, optim.Adam(model.parameters(), lr=0.001), epoch=epoch, verbose = False))
        torch.save(model.state_dict(), f"./Model Selection/models/epoch={epoch+load_epoch}/NN_{survey_code}.pt")
    elif model_type == "ratio-NN":
        if verbose:
            print(f"[{survey_code}] ratio-NN training")
        model = ratio_NN()
        for _ in range(epoch):
            loss_dict[model_type].append(model_train_func[model_type](survey_code, model, train_loader_ratio, test_loader, optim.Adam(model.parameters(), lr=0.001), epoch=epoch, verbose = False))
        torch.save(model.state_dict(), f"./Model Selection/models/epoch={epoch+load_epoch}/ratio-NN_{survey_code}.pt")
    elif model_type == "Meta":
        if verbose:
            print(f"[{survey_code}] Meta training")
        model_LSTM = meta_LSTM()
        model_NN = meta_NN()
        for _ in range(epoch):
            loss_dict[model_type].append(model_train_func[model_type](survey_code, model_LSTM, train_loader_LSTM, test_loader_LSTM, optim.Adam(model_LSTM.parameters(), lr=0.001), model_NN, optim.Adam(model_NN.parameters(), lr=0.001), epoch=epoch, verbose = False))
        torch.save(model_LSTM.state_dict(), f"./Model Selection/models/epoch={epoch+load_epoch}/Meta_LSTM_{survey_code}.pt")
        torch.save(model_NN.state_dict(), f"./Model Selection/models/epoch={epoch+load_epoch}/Meta_NN_{survey_code}.pt")
    elif model_type == "ratio-Meta":
        if verbose:
            print(f"[{survey_code}] ratio-Meta training")
        model_LSTM = meta_ratio_LSTM()
        model_NN = meta_ratio_NN()
        for _ in range(epoch):
            loss_dict[model_type].append(model_train_func[model_type](survey_code, model_LSTM, train_loader_ratio_LSTM, test_loader_ratio_LSTM, optim.Adam(model_LSTM.parameters(), lr=0.001), model_NN, optim.Adam(model_NN.parameters(), lr=0.001), epoch=epoch, verbose = False))
        torch.save(model_LSTM.state_dict(), f"./Model Selection/models/epoch={epoch+load_epoch}/ratio-Meta_LSTM_{survey_code}.pt")
        torch.save(model_NN.state_dict(), f"./Model Selection/models/epoch={epoch+load_epoch}/ratio-Meta_NN_{survey_code}.pt")
    
    #print(f"[{survey_code}] train complete.")
    return