import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

n = 274
bin_size = 9
use_white_list = False
white_list = [1, 2]

write_interval = True
to_bin = True
'''
toBin = [True, # Lambda
         False, # Rho
         False, # Mu
         False, # y_avg
         False, # y_median
         False, # y'_avg
         False, # y'_median
         False, # y'_1
         False, # y'_2
         False, # y'_3
         False, # y'_4
         False, # y'_5
         False, # order_avg
         False, # order_median
         False, # Meta-d'
         False  # M ratio
         ]
'''
factor_names = ['lambda', 'rho', 'mu', 'y_avg', 'y_median', "y'_avg", "y'_median", "y'_1", "y'_2", "y'_3", "y'_4", "y'_5", 'order_avg', 'order_median', "Meta-d'", 'M ratio']


#input: output of Processed_factors+MetaD.py
#source = open('./total_result_221004_class1.txt', 'r')
#source = open('./total_result_221025_class1_cut(Mratio0_1.5_rho0_mu0).txt', 'r')
source = open('./TOTAL_FACTORS_221115(n={}).txt'.format(n), 'r')

dest_data = [[] for i in range(16)]
# dest_range = [[1e+30, -1e+30] for i in range(11)]

survey_code_list = []

cnt = 0
for line in source.readlines():
    cnt += 1
    elements = (line[:-1] if line[-1] == '\n' else line).split('\t')
    survey_code_list.append(elements[0])
    elements = elements[1:] # remove '\n' at the end of line and split into values(still str).
    for idx in range(len(elements)):
        item = float(elements[idx])
        dest_data[idx].append(item)
        '''
        if item < dest_range[idx][0]:
            dest_range[idx][0] = item
        if item > dest_range[idx][1]:
            dest_range[idx][1] = item
        '''
print(cnt)
source.close()
#print('dest_data:', len(dest_data[0]))


#data_sorted = [sorted(dest_data[i]) for i in range(11)]
#data_pivot = [[data_sorted[idx][(((len(data_sorted[idx])-1)*(j+1))//bin_size)] for j in range(bin_size)] for idx in range(len(data_sorted))]
tmp = []
for i in range(bin_size):
    tmp.append(100*(i+1)/bin_size)

# make bin from dest_data
dest_bin = [[] for i in range(16)]
# [factor idx][participant idx]

if write_interval:
    interval = open('230201 n={} bin{} interval.txt'.format(n, bin_size), 'w')

for idx in range(16):
    data_pivot = np.percentile(dest_data[idx], tmp, interpolation='midpoint')
    # print("data_pivot: ", data_pivot)
    print("{} min: {}".format(factor_names[idx], min(dest_data[idx])))
    for pivot_idx in range(bin_size-1):
        print("{}_pivot for {}: ".format(factor_names[idx], pivot_idx+1), data_pivot[pivot_idx])
    #print("{}_pivot for 1: ".format(factor_names[idx]), data_pivot[0])
    #print("{}_pivot for {}: ".format(factor_names[idx], bin_size), data_pivot[bin_size-2])
    print("{} max: {}".format(factor_names[idx], max(dest_data[idx])))

    if write_interval:
        interval.write("{} min: {}\n".format(factor_names[idx], min(dest_data[idx])))
        for pivot_idx in range(bin_size-1):
            interval.write("{}_pivot for {}: {}\n".format(factor_names[idx], pivot_idx+1, data_pivot[pivot_idx]))
        interval.write("{} max: {}\n".format(factor_names[idx], max(dest_data[idx])))
        interval.write("\n")
    
    print()
    for j in range(len(dest_data[0])):
        if 8 <= j and j <= 12:
            if dest_data[idx][j] == 0:
                dest_bin[idx].append(1)
            elif dest_data[idx][j] < 1:
                dest_bin[idx].append(2)
            else:
                dest_bin[idx].append(3)
            continue
        
        for k in range(bin_size):
            if dest_data[idx][j] <= data_pivot[k] or k == (bin_size - 1):
                dest_bin[idx].append(k+1)
                break
            

if write_interval:
    interval.close()

print(np.shape(np.array(dest_bin)))
'''
for idx in range(len(factor_names)):
    plt.title("Distribution of {}".format(factor_names[idx]))
    plt.hist(dest_bin[idx], bins=[i for i in range(1, bin_size+2)])
    plt.savefig('./Histogram_{}.png'.format(factor_names[idx]))
    plt.clf()
'''
#dest = open('./total_result_bin{}_221108_masked_class1_cut(Mratio0_1.5_rho0_mu0).txt'.format(bin_size), 'w')
dest = open('./TOTAL_FACTORS_221115(n={})_bin{}.txt'.format(n, bin_size), 'w')

# todo : dest_data and dest_bin

for idx in range(len(dest_bin[0])):
    '''
    if idx == 164:
        for j in range(16):
            dest.write('\t')
        dest.write('\n')
    '''
    dest.write(survey_code_list[idx]+'\t')
    for j in range(16):
        if to_bin:
            if use_white_list:
                if dest_bin[j][idx] in white_list:
                    dest.write(str(dest_bin[j][idx]))
                else:
                    dest.write('.')
            else:
                dest.write(str(dest_bin[j][idx]))
        else:
            dest.write(str(dest_data[j][idx]))
        dest.write('\t')
    dest.write('\n')
dest.close()

