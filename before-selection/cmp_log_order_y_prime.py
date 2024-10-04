from math import isnan

yprime_file = open('./TOTAL_FACTORS_221115(n=274).txt', 'r')
log_file = open('./log_order_221115(n=274).txt', 'r')

yprime = [line.split('\t') for line in yprime_file.readlines()]
log_order = [line.split('\t') for line in log_file.readlines()]

y_idx = 0
log_idx = 0

y_len = len(yprime)
log_len = len(log_order)

print(y_len, log_len)

result_file = open(f'log_order_230201(n={274}).txt', 'w')

yprime = sorted(yprime)
log_order = sorted(log_order)

while y_idx < y_len and log_idx < log_len:
    if yprime[y_idx][0] != log_order[log_idx][0]:
        y_idx += 1
        continue

    #yp_line = yprime[y_idx]
    log_line = log_order[log_idx]
    #print(log_line)
    log_lines = []
    prev = 1
    for idx in range(len(log_line)):
        if log_line[idx] == '/':
            log_lines.append(log_line[prev:idx])
            prev = idx + 1
    log_lines.append(log_line[prev:-1])
    #print(log_lines)
    #levels = [ sum(list(map(float, log_line[1 + 9 * idx : 1 + 9 * (idx + 1)])))/9 for idx in range(5)] # avg
    #levels = [ sorted(list(map(float, log_line[1 + 9 * idx : 1 + 9 * (idx + 1)])))[4] for idx in range(5)] # median
    print(log_lines)
    #print(len(log_lines))
    #print()
    levels = [ sorted(list(map(float, log_lines[idx])))[len(log_lines[idx])//2] if len(log_lines[idx]) != 0 else 20 for idx in range(5)] # median
    #levels = [ sum(yprime[9 * idx : 9 * (idx + 1)]) for idx in range(5)] # median
    #print(levels)
    
    result_file.write(log_line[0] + '\t')
    for i in range(4):
        if levels[i] != float('inf') and levels[i] != float('-inf') and not isnan(levels[i]):
            result_file.write(str(levels[i]) + '\t')
        elif levels[i] == float('inf'):
            result_file.write(str(20) + '\t')
        elif levels[i] == float('-inf'):
            result_file.write(str(-20) + '\t')
        else:
            result_file.write(str(20) + '\t')

    if levels[4] != float('inf') and levels[4] != float('-inf') and not isnan(levels[4]):
        result_file.write(str(levels[4]) + '\t')
    elif levels[4] == float('inf'):
        result_file.write(str(20) + '\t')
    elif levels[4] == float('-inf'):
        result_file.write(str(-20) + '\t')
    else:
        result_file.write(str(20) + '\t')

    result_file.write('\n')
    y_idx += 1
    log_idx += 1

result_file.close()