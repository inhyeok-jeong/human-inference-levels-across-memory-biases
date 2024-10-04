import numpy as np
import scipy.optimize as opt
import os


# data lodaing part

#file_dir = "./data/A1JTMH37C1UH7J.txt"

analyze_kbc = True
analyze_pros = False
use_mat_pros = True
analyze_meta = False

file_names = []

def read_file(file_dir, print_id = False):
    global kbc_lines, prospect_lines, meta_lines
    # data reading part

    file = open(file_dir, 'r')
    txt = file.read()
    lines = list(txt.split('&'))

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
    
    return kbc_lines, prospect_lines, meta_lines

# data analysis part

def kbc():
    global kbc_lines
    # [[y], [y'], [orders]]
    kbc_result = [ [], [], [[] for i in range(5)]]
    
    for contest_idx in range(5):
        reward = [0 for i in range(5)]
        for round_idx in range(10):
            #if round_idx < 10:
                #kbc_result[1][contest_idx].append(np.log(kbc_lines[11*contest_idx+9][round_idx+1] / (np.sum(kbc_lines[11*contest_idx:11*contest_idx+10, round_idx]/10))) / np.log(2/3))
            ans = (np.sum(kbc_lines[11*contest_idx:11*contest_idx+10, round_idx])/10 * 2/3)
            diff = np.round(np.abs(np.array(kbc_lines[11*contest_idx:11*contest_idx+10, round_idx]) - ans), 2)
            # print('diff')
            # print(diff)
            m = min(diff)
            for i in range(9):
                if diff[i] == m and i % 2 == 0:
                    reward[i//2] += 1
        # print('Reward of {}:'.format(contest_idx))
        # print(reward)
        y = sum([(i+1)*reward[i] for i in range(5)])/sum(reward) - 1
        y_prime = ( 10 * y - 15 ) / 5
        kbc_result[0].append(y)
        kbc_result[1].append(y_prime)
        # print('y:{}'.format(kbc_result[0][-1]))

    for contest_idx in range(5):
        for round_idx in range(1, 10):
            temp = kbc_lines[11*contest_idx+9, round_idx]
            temp2 = kbc_lines[11*contest_idx+9, round_idx-1]
            '''
            for i in range(len(temp)):
                kbc_result[1][contest_idx].append(np.log(temp[i]/temp2[i]) / np.log(2/3))
            '''
            # print(temp, temp2)
            kbc_result[2][contest_idx].append(np.log(temp/temp2) / np.log(2/3))

    return kbc_result

# prospect_factor = [1, 2, 1] = [lambda, rho, mu]

def u(x, prospect_factor):
    if x > 0:
        return x ** prospect_factor[1]
    else:
        return - 1 * prospect_factor[0] * (-1 * ((-1*x)**prospect_factor[1]))

def prospect(prospect_factor):
    global prospect_lines
    x = prospect_factor
    f = 0
    
    for line in prospect_lines:
        # line: [gain, loss, sure, select(y), result]
        # matlab: [sure, gain, loss, select(y)]
        xb = (x[0]) * ( 0.5*(line[0]**(x[2])) - (0.5*(x[1])*abs(line[1])) - (line[2]**(x[2])) )

        # theGreatStatsby article
        #xb = (0.5*u(line[0], x)) + (0.5*u(-1*line[1], x)) - u(line[2], x)

        if line[3] == 1: # if selected gamble
            f = f + np.log(1/(1+np.exp(-1*xb)))
            #f = f + np.log(1 / (1+np.exp(-1*x[2]*xb)))
        elif line[3] == 0: # if selected sure
            f = f + np.log(1-1/(1+np.exp(-1*xb)))
            #f = f - np.log((1 - 1 / (1+np.exp(-1*x[2]*xb))))
            
    f = f * -1
    return f

    #data = lines[:, ]

def read_prospect(file_dir):
    file = open('./'+file_dir, 'r')
    prospect_result_lines = []
    for line in file.readlines():
        prospect_result_lines.append(line[:-1])
    
    file.close()    
    return prospect_result_lines

def meta(meta_factor):
    pass

# data visualization part

# Main 

if __name__ == '__main__':

    kbc_result = []
    prospect_result = []
    meta_result = []

    bound = opt.Bounds([0, 0, 0], [10000, 1.5, 100])

    corr = [[[] for j in range(15)] for i in range(3)] # [prospect_idx][kbc_idx]
    
    for file_name in os.listdir("accepted_data"):
        read_file('./accepted_data/'+file_name)
        file_names.append(file_name[:-4])

        if analyze_kbc:
            kbc_result.append(kbc())
        if analyze_pros:
            prospect_result.append(opt.fmin(func=prospect, x0=[1, 2, 1], xtol=1e-1, maxiter=1e+4)) # checked it works in 2022/08/29

    if analyze_kbc:
        kbc_result = np.array(kbc_result)

    # Calc average and median of y and y'.
    kbc_y_avg = []
    kbc_y_median = []
    kbc_y_prime_avg = []
    kbc_y_prime_median = []
    kbc_y_primes = []
    kbc_order_avg = []
    kbc_order_median = []

    # [[y], [y'], [orders]]

    for idx in range(len(kbc_result)):
        kbc_y_line = kbc_result[idx][0]
        kbc_y_avg.append(sum(kbc_y_line)/5)
        kbc_y_median.append(sorted(kbc_y_line)[2])
        kbc_y_prime_line = kbc_result[idx][1]
        kbc_y_prime_avg.append(sum(kbc_y_prime_line)/5)
        kbc_y_prime_median.append(sorted(kbc_y_prime_line)[2])
        for j in range(5):
            kbc_y_primes.append(kbc_y_prime_line[j])
        
    for idx in range(len(kbc_result)):
        kbc_line = kbc_result[idx][2]
        kbc_order_avg.append(sum([sum(item)/5 for item in kbc_line])/5)
        kbc_order_list = kbc_line[0] + kbc_line[1] + kbc_line[2] + kbc_line[3] + kbc_line[4]
        kbc_order_median.append(sorted(kbc_order_list)[24])

    # prospect_result[idx]: [lambda, rho, mu]
    if analyze_pros:
        prospect_result = np.array(prospect_result)

    if use_mat_pros:
        prospect_result_lines = read_prospect('Prospect_Result_221112.txt')
        #print(prospect_result_lines[0])

    result_file = open('processed_factors_221112.txt', 'w')
    for i in range(len(prospect_result_lines)):
        result_file.write(prospect_result_lines[i]+'\t')
        result_file.write(str(kbc_y_avg[i])+'\t')
        result_file.write(str(kbc_y_median[i])+'\t')
        result_file.write(str(kbc_y_prime_avg[i])+'\t')
        result_file.write(str(kbc_y_prime_median[i])+'\t')
        for j in range(5): # To check each y' and Mratio corr (221025 update)
            result_file.write(str(kbc_y_primes[5*i+j])+'\t')
        result_file.write(str(kbc_order_avg[i])+'\t')
        result_file.write(str(kbc_order_median[i]))

        result_file.write('\n')

    result_file.close()
    
    '''
    file_name = '1008A202215U1111.txt'
    read_file('./data/'+file_name)
    
    kbc_result.append(kbc())
    prospect_result.append( opt.fmin( func=prospect, x0=np.array([1, 2, 1]) ) ) # checked it works in 2022/09/14
    #prospect_result.append( opt.minimize( prospect, x0=np.array([1, 2, 1], dtype=np.float64), bounds=bound ) ) # checked it works in 2022/09/14
    
    # bounds=[[0, 10000], [0, 1.5], [0, 100]]
    '''    
    
    """
    for i in range(3):
        for j in range(5):
            corr[i][j]=np.corrcoef(prospect_result[:,i], np.array(kbc_result[:, 0, j], dtype=np.float64))[0][1]
        '''
        for j in range(5):
            for k in range(9):
                corr[i][j+5].append(np.corrcoef(prospect_result[:,i], np.array(kbc_result[:, 1, k], dtype=np.float64)))
        '''
    print(prospect_result)
    # print(corr)
    """
