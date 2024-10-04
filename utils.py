import os
import numpy as np

dest_dir = '2023-11-23'
epoch = 30

def read_file(file_dir, print_id = False):
    global kbc_lines, prospect_lines, meta_lines
    # data reading part

    file = open(file_dir, 'r')
    txt = file.read()
    lines = list(txt.split('&'))

    survey_code = lines[0]

    kbc_lines = lines[3].split('/')

    new_lines = [ item.split('_') for item in kbc_lines]
    if new_lines[-1] == ['']:
        new_lines = new_lines[:-1]
    kbc_lines = new_lines
    new_lines = []
    
    for item_list in kbc_lines:
        for item in item_list:
            new_lines.append(list(map(float, item.split(','))))
    kbc_lines = new_lines
    kbc_lines = np.array(kbc_lines)
    
    prospect_lines = lines[4]
    prospect_lines = prospect_lines.split('/')[:-1]
    new_lines = []
    for line in prospect_lines:
        new_lines.append(list(map(float, line.split(','))))

    # print(new_lines)
    prospect_lines = np.array(new_lines)

    meta_lines = lines[5]
    meta_lines = meta_lines.split('/')[:-1]
    new_lines = []
    for line in meta_lines:
        new_lines.append(list(map(float, line.split(','))))

    meta_lines = new_lines

    file.close()

    if print_id:
        print(lines[1])
    
    return survey_code, kbc_lines, prospect_lines, meta_lines

def read_behavior_data(data_dir: str, verbose=True):
    data_dict = {}

    # generate whitelist(n=274 or n=297)
    whitelist_dir = './TOTAL_FACTORS_221115(n=274).txt'
    whitelist = []

    for line in open(whitelist_dir, 'r').readlines():
        data = line.split('\t')
        whitelist.append(data[0])
        
    # read data
    for file_dir in os.listdir(data_dir):
        if verbose:
            print(file_dir)
        if not file_dir.split("/")[-1][:-4] in whitelist:
            continue
        survey_code, kbc_lines, _, _ = read_file(data_dir + '/' + file_dir)
        data_dict[survey_code] = kbc_lines
    print(f"read {len(data_dict.keys())} behavior data.")
    return data_dict

def compose_behavior_data(individual_raw_data):
    # raw data: list consist of 55 lists.
    # raw data: [10 agents, 1 response time continue 5 times]
    # raw data element(agent): [10 actions]
    behavior_data = []
    for contest_idx in range(5):
        contest_list = []
        for round_idx in range(10):
            round_list = []
            for agent_idx in range(10):
                round_list.append(individual_raw_data[11 * contest_idx + agent_idx][round_idx])
            contest_list.append(round_list)
        behavior_data.append(contest_list)

    # return value: list consist of 5 lists(5 contests).
    return behavior_data

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


def read_match_dict(dir_name, epoch, target_combination_idx):
    with open(f"./Model Selection/output/{dir_name}/epoch={epoch}/value_log.txt", 'r') as file:
        _ = file.readline()
        _ = file.readline()
        line = file.readline()[:-1]
        line = file.readline()[:-1]
        match_dict = eval(line)
    return match_dict[target_combination_idx]

def read_factors():
    factors = {}
    factor_names = ["Lambda", "Rho", "Mu", "y_avg", "y_med", "y'_avg", "y'_med", "y'_1", "y'_2", "y'_3", "y'_4", "y'_5", "log_avg", "log_med", "Meta-d'", "M-ratio"]
    for name in factor_names:
        factors[name] = {}

    with open("./TOTAL_FACTORS_221115(n=274).txt", "r") as file:
        for line in file.readlines():
            elements = line.split("\t")
            if elements[-1] == "":
                elements = elements[:-1]
            survey_code = elements[0]
            for idx in range(16):
                factors[factor_names[idx]][survey_code] = float(elements[1 + idx])
    return factors

def read_competition_data(dest_dir = '2023-11-23', epoch = 30):
    sample_data = []
    f = open(f"./Model Selection/output/{dest_dir}/epoch={epoch}/competition.txt", "r")
    for _ in range(822):
        sample_one = []
        sample_one.append(f.readline()[:-1])
        sample_one.append(f.readline()[:-1])
        sample_one.append(f.readline()[:-1])
        sample_data.append(sample_one)
    f.close()

    subject_data = dict()

    sample_num_dict = {}
    for sample_idx in range(822):
        population = eval(sample_data[sample_idx][0])
        for survey_code in population:
            sample_num_dict[survey_code] = 0

    for sample_idx in range(822):
        population = eval(sample_data[sample_idx][0])
        #print(population)
        sample_data[sample_idx][1]
        actions_log_array = np.array(eval(sample_data[sample_idx][1]))
        rewards_log_array = np.array(eval(sample_data[sample_idx][2]))
        for agent_idx in range(10):
            survey_code = population[agent_idx]
            sample_num_dict[survey_code] += 1

            if survey_code not in subject_data.keys():
                subject_data[survey_code] = {
                    'actions': [],
                    'rewards': [],
                    'avg': []
                }
            subject_data[survey_code]['actions'].append(actions_log_array[:, agent_idx])
            subject_data[survey_code]['rewards'].append(rewards_log_array[:, agent_idx])
            subject_data[survey_code]['avg'] = [avg(actions_log_array[round_idx]) for round_idx in range(len(actions_log_array))]
    return subject_data

def avg(x):
    return sum(x) / len(x)

def median(L):
    L = sorted(L)
    if len(L) % 2 == 0:
        return (L[len(L) // 2 - 1] + L[len(L) // 2]) / 2
    return L[len(L) // 2]

def calc_p(last_actions, action):
    #print(f"actions: {last_actions}")
    #print(f"avg of actions: {avg(last_actions)}, rewarded action: {action}")
    #print(f"p: {action / avg(last_actions)}")
    return action / avg(last_actions)

def convert_type2color(model_type):
    if model_type == 'NN':
        return 'red'
    elif model_type == 'ratio-NN':
        return 'orange'
    elif model_type == 'Meta':
        return 'green'
    elif model_type == 'ratio-Meta':
        return 'blue'
    elif model_type == 'KNN':
        return 'purple'
    elif model_type == "Meta/high":
        return 'forestgreen'
    elif model_type == "Meta/low":
        return 'greenyellow'
    
def convert_model2name(model_type):
    if model_type == "Meta":
        return "Episodic"
    elif model_type == "ratio-NN":
        return "Working"
    elif model_type == "KNN":
        return "Habit"
    return None

def convert_models2name(model_types):
    return list(map(convert_model2name, model_types))

def convert_model2color(model_type):
    if model_type == 'Meta':
        return 'red'
    elif model_type == 'ratio-NN':
        return 'green'
    elif model_type == 'KNN':
        return 'blue'

def convert_name2color(model_name):
    if model_name == 'Episodic':
        return 'red'
    elif model_name == 'Working':
        return 'green'
    elif model_name == 'Habit':
        return 'blue'
