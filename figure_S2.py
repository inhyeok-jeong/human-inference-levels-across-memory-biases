from utils import *
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statannot import add_stat_annotation
import argparse

def get_age_group(age):
    i = age // 10
    if i > 6:
        i = 6
    return f"{i}0" + ("+" if i == 6 else "'s")

def priority(row):
    # row: [survey_code, y'_avg, sex, age, edu]
    p = 0
    if row[2] == "female":
        p = 100
    else:
        p = 200

    p += int(row[3][0]) * 10

    if row[4] == "High school graduation":
        p += 1
    elif row[4] == "Bachelor (B.A., B.S.)":
        p += 2
    elif row[4] == "Master (M.A., M.S.)":
        p += 3
    return p

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--font", default="Calibri")
    parser.add_argument("--font_size", type=int, default=16)
    parser.add_argument("--output_dir", default="output")

    args = parser.parse_args()
    plt.rcParams['font.family'] = args.font
    plt.rcParams['font.size'] = args.font_size
    output_dir = "/".join([args.output_dir, "figS2", f"{args.font}_{args.font_size}"])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    factors = read_factors()
    factors["sex"] = {}
    factors["age"] = {}
    factors["edu"] = {}

    ages = []

    sex = {
        "Male": 0,
        "Female": 0,
    }

    edu = {}

    

    for survey_code in factors["y'_avg"].keys():
        with open(f"./raw data/DATA/accepted_data_total/{survey_code}.txt") as f:
            text = f.read()
            contents = text.split("&")[2].split("/")
            ages.append(int(contents[0]))
            contents[1] = contents[1].lower()
            factors["age"][survey_code] = get_age_group(int(contents[0]))
            factors["sex"][survey_code] = contents[1][0].upper() + contents[1][1:]
            if contents[1][0].upper() + contents[1][1:] in sex.keys():
                sex[contents[1][0].upper() + contents[1][1:]] += 1
            else:
                sex[contents[1][0].upper() + contents[1][1:]] = 1
            if contents[2] == "masters":
                contents[2] = "Master (M.A., M.S.)"
            elif contents[2] == "Bachelor's":
                contents[2] = "Bachelor (B.A., B.S.)"
            
            factors["edu"][survey_code] = contents[2]
            if contents[2] in edu.keys():
                edu[contents[2]] += 1
            else:
                edu[contents[2]] = 1
    
    print(ages) # [25, 38, 28, 42, 26, 60, 46, 33, 34, 30, 32, 36, 35, 30, 32, 25, 26, 67, 36, 38, 25, 32, 25, 32, 26, 26, 39, 29, 24, 25, 28, 39, 38, 32, 43, 59, 38, 40, 35, 25, 35, 26, 28, 23, 34, 25, 61, 39, 32, 51, 25, 33, 56, 35, 28, 31, 48, 55, 48, 27, 33, 24, 31, 30, 48, 33, 48, 27, 37, 26, 48, 49, 30, 33, 32, 28, 33, 25, 43, 26, 23, 33, 32, 48, 48, 48, 35, 35, 33, 31, 37, 32, 35, 48, 53, 33, 31, 24, 33, 38, 26, 25, 33, 26, 48, 25, 25, 25, 33, 48, 37, 25, 25, 25, 25, 52, 56, 49, 39, 40, 66, 35, 55, 25, 37, 25, 25, 25, 25, 25, 25, 33, 25, 54, 40, 25, 25, 54, 25, 25, 25, 25, 57, 25, 25, 54, 25, 33, 25, 25, 25, 25, 25, 27, 25, 41, 34, 51, 27, 25, 41, 25, 22, 25, 30, 25, 25, 32, 26, 35, 59, 67, 25, 66, 27, 62, 24, 35, 25, 33, 28, 64, 25, 38, 32, 30, 32, 25, 29, 33, 38, 43, 25, 66, 25, 61, 30, 57, 32, 48, 55, 38, 32, 25, 50, 22, 25, 44, 31, 33, 25, 37, 25, 25, 34, 45, 32, 49, 33, 33, 34, 48, 38, 40, 30, 30, 35, 45, 34, 28, 40, 52, 56, 56, 25, 56, 57, 48, 25, 42, 25, 25, 30, 32, 24, 55, 47, 53, 42, 42, 30, 34, 35, 24, 35, 68, 49, 52, 32, 48, 38, 29, 35, 33, 25, 27, 33, 26, 33, 25, 25, 25, 25, 38]
    print(sex) # {'male': 137, 'female': 137}
    print(edu) # {'Master (M.A., M.S.)': 86, 'Bachelor (B.A., B.S.)': 177, 'masters': 1, 'High school graduation': 9, "Bachelor's": 1}
    print(len(ages))

    print(min(ages), max(ages))
    age_cnt = [0 for _ in range(8)]
    for age in ages:
        age_cnt[age // 10] += 1
    age_cnt[6] = age_cnt[6] + age_cnt[7]
    age_cnt = age_cnt[:-1]
    print(age_cnt)

    df_data = []
    for survey_code in factors["y'_avg"].keys():
        df_data.append([survey_code, factors["y'_avg"][survey_code], factors["sex"][survey_code], factors["age"][survey_code], factors["edu"][survey_code]])
    
    # filter
    df_data.sort(key=lambda x: priority(x))

    df = pd.DataFrame(data=df_data, columns=["survey_code", "y'_avg", "sex", "age", "edu"])

    print("====")
    # y'_avg
    print(max(factors["y'_avg"].values()), min(factors["y'_avg"].values()))
    plt.hist(factors["y'_avg"].values(), bins=[0.5 * (i - 6) for i in range(0, 12)])
    #plt.hist(factors["y'_avg"].values())
    plt.xlabel("y'_avg")
    plt.tight_layout()
    #plt.xticks([0.5 * i for i in range(0, 7)])
    plt.savefig(f"{output_dir}/hist.png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/hist.pdf", bbox_inches='tight')

    # age
    plt.clf()
    plt.hist(ages)
    plt.ylabel("participants")
    plt.xlabel("ages")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/age.png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/age.pdf", bbox_inches='tight')

    # sex
    plt.clf()
    plt.pie([sex['Female'], sex['Male']], labels=['Female(137)', 'Male(137)'], colors=["red", "blue"])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sex.png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/sex.pdf", bbox_inches='tight')

    # education
    plt.clf()
    plt.pie([edu['High school graduation'], edu['Bachelor (B.A., B.S.)'], edu['Master (M.A., M.S.)']], startangle=45, labels = ["High school graduation(9)", "Bachelor(178)", "Master(87)"])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/edu.png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/edu.pdf", bbox_inches='tight')

    # age pie chart
    plt.clf()
    plt.pie(age_cnt[2:], labels=[f"{i}0" + ("+" if i == 6 else "'s") + f"({age_cnt[i]})" for i in range(2, 7)])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/age-pie.png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/age-pie.pdf", bbox_inches='tight')

    plt.clf()
    ax = sns.boxplot(data=df, x="sex", y="y'_avg", palette={"Female":"red", "Male":"blue"})
    add_stat_annotation(ax, data=df, x=f"sex", y=f"y'_avg", 
                        box_pairs=[("Female", "Male")],
                        test="t-test_welch", text_format="star", loc="inside", 
                        pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, ""]])
    plt.xlabel("")
    plt.ylabel("$y'_{avg}$")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/y'_avg-sex.png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/y'_avg-sex.pdf", bbox_inches='tight')

    '''plt.clf()
    ax = sns.boxplot(data=df, x="edu", y="y'_avg")
    edu_keys = edu.keys()
    box_pairs = []
    for key1 in edu_keys:
        for key2 in edu_keys:
            if (key2, key1) in box_pairs or key1 == key2:
                continue
            box_pairs.append((key1, key2))
    add_stat_annotation(ax, data=df, x=f"edu", y=f"y'_avg", 
                        box_pairs=box_pairs,
                        test="t-test_welch", text_format="star", loc="inside", 
                        pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, ""]])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/y'_avg-edu.png", bbox_inches='tight')'''

    plt.clf()
    ax = sns.boxplot(data=df, x="age", y="y'_avg")
    age_keys = [get_age_group(age) for age in [20, 30, 40, 50, 60]]
    box_pairs = []
    for key1 in age_keys:
        for key2 in age_keys:
            if (key2, key1) in box_pairs or key1 == key2:
                continue
            box_pairs.append((key1, key2))
    add_stat_annotation(ax, data=df, x=f"age", y=f"y'_avg", 
                        box_pairs=box_pairs,
                        test="t-test_welch", text_format="star", loc="inside", 
                        pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, ""]])
    plt.xlabel("")
    plt.ylabel("$y'_{avg}$")
    plt.ylim((-3,3))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/y'_avg-age.png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/y'_avg-age.pdf", bbox_inches='tight')