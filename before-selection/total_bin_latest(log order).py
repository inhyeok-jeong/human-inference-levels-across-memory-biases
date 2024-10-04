import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

n = 274
bin_size = 9
use_white_list = False
white_list = [1, 2]

write_interval = True
to_bin = True
factor_names = ['level-1', 'level-2','level-3', 'level-4','level-5']


#input: output of Processed_factors+MetaD.py

source = open('./log_order_230201(n=%d).txt'.format(n), 'r')

dest_data = [[] for i in range(5)]
# dest_range = [[1e+30, -1e+30] for i in range(11)]

survey_code_list = []
src = []

cnt = 0
for line in source.readlines():
    elements = (line[:-1] if line[-1] == '\n' else line).split('\t')
    print(elements)

    if 'inf' in elements or '-inf' in elements or 'nan' in elements:
        #continue
        for i in range(len(elements)):
            if elements[i] == 'inf' or elements[i] == 'nan':
                elements[i] = '20'
            elif elements[i] == '-inf':
                elements[i] = '-20'
    src.append(elements)

    cnt += 1
    survey_code_list.append(elements[0])
    elements = elements[1:] # remove '\n' at the end of line and split into values(still str).
    for idx in range(5):
        item = float(elements[idx])
        #print(idx - 1)
        dest_data[idx].append(item)
        '''
        if item < dest_range[idx][0]:
            dest_range[idx][0] = item
        if item > dest_range[idx][1]:
            dest_range[idx][1] = item
        '''
print(cnt)
source.close()

source = open('./231202 result.txt'.format(n), 'w')
p = 0
for line in src:
    source.write(survey_code_list[p])
    for e in line:
        source.write(e + '\t')
    source.write('\n')
    p += 1

source.close()

#print('dest_data:', len(dest_data[0]))


#data_sorted = [sorted(dest_data[i]) for i in range(11)]
#data_pivot = [[data_sorted[idx][(((len(data_sorted[idx])-1)*(j+1))//bin_size)] for j in range(bin_size)] for idx in range(len(data_sorted))]
tmp = []
for i in range(bin_size):
    tmp.append(100*(i+1)/bin_size)

# make bin from dest_data
dest_bin = [[] for i in range(5)]
# [factor idx][participant idx]

if write_interval:
    interval = open('231202 n={} bin{} interval.txt'.format(cnt, bin_size), 'w')

#print(tmp)
for idx in range(5):
    #print(dest_data[idx])
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
        if dest_data[idx][j] == 'inf':
            dest_bin[idx].append('.')
            continue
        if dest_data[idx][j] == 'nan':
            dest_bin[idx].append('.')
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
dest = open('./log_order_221115(n={})_bin{}.txt'.format(n, bin_size), 'w')

# todo : dest_data and dest_bin

for idx in range(len(dest_bin[0])):
    dest.write(survey_code_list[idx]+'\t')
    for j in range(5):
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

