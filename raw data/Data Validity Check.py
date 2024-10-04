import os
import numpy as np
import pandas as pd

notice = [[] for i in range(10)]
'''
[0] : total
[1] : Selected same thing consecutively more than 10 or 30 times in Prospect theory.
[2] : Selected same direction consecutively more than 10 times
2_1 : Selected same direction consecutively more than 10 times, accuracy >= 50% for the interval.
[5] : input confidence=50% more than half during Metacognition
[6] : ([2] && !2_1) || ([5]) (Metacognition)
[7] : [1] || [6] (Prospect Theory or Metacognition)
[8] : Selected big value more than three times in short time during Keynesian Beauty Contest
[9] : [6] || [8] (Metacognition or Keynesian Beauty Contest)
'''
accept = []
accept_id = []
threshold = 100
frequency = []
notice2_1 = []
cnt = 0

# "data_221108"
# "indiv"
for file_name in os.listdir("data_221112"):
    with open(os.path.join("data_221112", file_name), 'r') as f:
        txt = f.read()
        # print(txt)
        flag = False
        flag1 = False
        flag2 = False
        flag2_1 = True
        flag3 = False
        raw_list = txt.split('&')
        '''
         [0]: Survey Code
         [1]: WorkerId
         [2]: Info
         [3]: KBC
             [
              [Agent[0], Agent[1], Agent[2], Agent[3], Agent[4], Agent[5], Agent[6], Agent[7], Agent[8], Agent[9], ResponseTime(ms)],
              [Agent[0], Agent[1], Agent[2], Agent[3], Agent[4], Agent[5], Agent[6], Agent[7], Agent[8], Agent[9], ResponseTime(ms)],
              [Agent[0], Agent[1], Agent[2], Agent[3], Agent[4], Agent[5], Agent[6], Agent[7], Agent[8], Agent[9], ResponseTime(ms)],
              [Agent[0], Agent[1], Agent[2], Agent[3], Agent[4], Agent[5], Agent[6], Agent[7], Agent[8], Agent[9], ResponseTime(ms)],
              [Agent[0], Agent[1], Agent[2], Agent[3], Agent[4], Agent[5], Agent[6], Agent[7], Agent[8], Agent[9], ResponseTime(ms)]
             ]
         [4]: Prospect Theory
              [ [Gamble Gain, Gamble Lose, Sure, Choice, Result] ]
         [5]: Metacognition
              [ [isExample, LeftDots, RightDots, Choice, Confidence, ResponseTime] ]
        '''
        '''
        for i in range(6):
            print(raw_list[i])
        '''
        raw_list[2] = raw_list[2].split('/')
        raw_list[3] = raw_list[3].split('/')
        raw_list[4] = raw_list[4].split('/')
        raw_list[5] = raw_list[5].split('/')

        new_list = [ item.split('_') for item in raw_list[3] ]
        raw_list[3] = new_list
        if raw_list[3][-1] == ['']:
            raw_list[3] = raw_list[3][:-1]
        new_list = []
        for item_list in raw_list[3]:
            for item in item_list:
                new_list.append(list(map(float, item.split(','))))
        raw_list[3] = new_list
        
        for i in range(4, 6):
            new_list = []
            for item in raw_list[i]:
                if i == 2:
                    new_list.append(item.split(','))
                else:
                    x = item.split(',')
                    if x == ['']:
                        continue
                    # print(x)
                    if x == ['null', 'null', 'null']:
                        new_list.append(x)
                        continue
                    new_list.append(list(map(float, x)))
            raw_list[i] = new_list

        raw_list[5] = raw_list[5][8:]

        raw_list = np.array(raw_list)

        invalid = False
        for idx in range(3, len(raw_list)):
            if len(raw_list[idx]) == 0 or raw_list[idx] == ['null'] or raw_list[idx] == ['']:
                invalid = True
        if invalid:
            print(raw_list)
            print('skipped {} since there is null'.format(file_name))
            continue
        
        
        # print(raw_list)
        #print(np.shape(raw_list[4]))
        # (1)
        #print(np.array(raw_list[4]))
        for i in range(len(raw_list[4])-30):
            #print(np.array(raw_list[4])[i:i+10, 3])
            x = np.sum(np.array(raw_list[4])[i:i+30, 3])
            if x == 0 or x == 30 or x == 60:
                flag1 = True
        # (2)
        for i in range(len(raw_list[5])-10):
            x = np.sum(np.array(raw_list[5])[i:i+10, 3])
            if x == 0 or x == 10:
                flag2 = True
                y = np.array(raw_list[5])[i:i+10, 3]
                L = np.array(raw_list[5])[i:i+10, 1]
                R = np.array(raw_list[5])[i:i+10, 2]
                t = np.array([0 if L[j] > R[j] else 1 for j in range(10)])
                cnt_ans = 0
                '''
                print('========')
                print(y)
                print(t)
                print('========')
                '''
                for j in range(10):
                    if y[j] == t[j]:
                        cnt_ans += 1
                # print(cnt_ans)
                if flag2_1 and cnt_ans >= 5: # 같은 선택을 10회 반복했지만 정답률이 50% 이상이라면 제외
                    flag2_1 = True
                else:
                    flag2_1 = False

        # (3)
        for i in range(len(raw_list[5])-10):
            small_flag = True
            for j in range(10):
                if raw_list[5][i+j][4] != raw_list[5][i+j+1][4]:
                    small_flag = False
            if small_flag:
                flag3 = True

        if flag1 or flag2 or flag3:
            flag = True

        if flag:
            notice[0].append(raw_list[0])
        if flag1:
            notice[1].append(raw_list[0])
        if flag2:
            notice[2].append(raw_list[0])
            if flag2_1:
                notice2_1.append(raw_list[0])
        if flag3:
            notice[3].append(raw_list[0])

        confidence = np.array(raw_list[5])[:, 4]
        #print(confidence)
        #print(len(confidence))
        no_use_confidence = True
        for i in confidence:
            if i != 50:
                no_use_confidence = False
                break
        if no_use_confidence:
            notice[4].append(raw_list[0])

        # fre = np.bincount(confidence)
        fre, bins = np.histogram(confidence, np.arange(0, 100, 2))
        frequency.append(fre)

        flag5 = fre[25] >= threshold

        if flag5:
            notice[5].append(raw_list[0])

        flag6 = (flag2 and not flag2_1) or flag5

        if flag6:
            notice[6].append(raw_list[0])

        flag7 = flag1 or flag6
        if flag7:
            notice[7].append(raw_list[0])

        flag8 = False
        for i in range(5): # 5
            for j in range(10-2):
                # np.shape(raw_list[3]) = (55, 10)
                # agent[0~9], ResponseTime
                # print(raw_list[3])
                
                if (raw_list[3][11*i+9][j] > 1
                    and raw_list[3][11*i+9][(j+1)] > 1
                    and raw_list[3][11*i+9][(j+2)] > 1):
                    if (raw_list[3][11*i+10][j] < 500
                        and raw_list[3][11*i+10][(j+1)] < 500
                        and raw_list[3][11*i+10][(j+2)] < 500):
                        flag8 = True
                    '''
                if ( raw_list[3][11*i+9][j] == raw_list[3][11*i+9][(j+1)]
                    and raw_list[3][11*i+9][(j+1)] == raw_list[3][11*i+9][(j+2)] ):
                '''
                '''
                if ( raw_list[3][11*i+9][j] == raw_list[3][11*i+9][(j+1)]
                    and raw_list[3][11*i+9][(j+1)] == raw_list[3][11*i+9][(j+2)]
                    and raw_list[3][11*i+9][j] != 0):
                    '''
                    

        if flag8:
            notice[8].append(raw_list[0])

        flag9 = flag6 or flag8
        if flag9:
            notice[9].append(raw_list[0])

        '''
        if not flag9:
            accept.append(raw_list[0])
            accept_id.append(raw_list[1])
            result_file = open("./accepted data/"+raw_list[1]+".txt", 'w')
            result_file.write(txt)
            result_file.close()
        '''
        flag10 = False
        cnt_contest = 0
        for i in range(5): # 5
            cnt_round = 0
            for j in range(10):
                # np.shape(raw_list[3]) = (55, 10)
                # agent[0~9], ResponseTime
                # print(raw_list[3])
                
                if (raw_list[3][11*i+9][j] >= 70):
                    cnt_round += 1

            if cnt_round >= 2:
                cnt_contest += 1
        if cnt_contest >= 1:
            flag10 = True

        if not flag9 and not flag10:
        #if True:
            accept.append(raw_list[0])
            accept_id.append(raw_list[1])
            
            result_file = open("./accepted_data/"+raw_list[0]+".txt", 'w')
            result_file.write(txt)
            result_file.close()
            
        '''
        print(flag2)
        print(flag2_1)
        print(flag5)
        print(fre)
        print(flag6)
        print(flag8)
        '''
        
        cnt += 1
        print("{} Done.".format(cnt))

print(notice)
print(len(accept))
