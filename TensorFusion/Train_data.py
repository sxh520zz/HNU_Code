# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:58:02 2019

@author: shixiaohan
"""

import pickle
import math
import numpy as np
# reload a file to a variable
with open('../IEM_Feature/Text_data.pickle', 'rb') as file:
    train_org_data_map = pickle.load(file)

def Train_data(train_map):
    num = 0
    input_traindata_x = []
    input_traindata_y = []
    input_traindata_z = []
    input_traindata_y_1 = []
    for i in range(len(train_map)):
        input_trainlabel_2 = []
        input_traindata_3 = []
        input_traindata_4 = []
        input_traindata_5 = []
        for x in range(len(train_map[i])):
            input_trainlabel_2.append(train_map[i][x]['label'])
            input_traindata_3.append(train_map[i][x]['trad_data'])
            input_traindata_4.append(train_map[i][x]['id'])
            input_traindata_5.append(train_map[i][x]['transcr_data'])
            num = num+1
        input_traindata_x.append(input_trainlabel_2)
        input_traindata_y.append(input_traindata_3)
        input_traindata_z.append(input_traindata_4)
        input_traindata_y_1.append(input_traindata_5)
    print(num)


    label_list= [1,2,3,4,5]
    num = 0
    traindata_1 = []
    for i in range(len(input_traindata_z)):
        input_traindata_1_1 = []
        for x in range(len(input_traindata_z[i])):
            a = {}
            if (input_traindata_x[i][x] in label_list):
                if (input_traindata_x[i][x] == 5):
                    input_traindata_x[i][x] = 2
                a['label_emotion'] = int(input_traindata_x[i][x] - 1)
                a['trad_data'] = input_traindata_y[i][x]
                a['transcr_data'] = input_traindata_y_1[i][x]
                a['id'] = input_traindata_z[i][x]
                a['section_id'] = input_traindata_z[i][x][4]
                input_traindata_1_1.append(a)
                num = num + 1
        traindata_1.append(input_traindata_1_1)
    print(num)
    data_1 = []
    data_2 = []
    data_3 = []
    data_4 = []
    data_5 = []

    for i in range(len(traindata_1)):
        for j in range(len(traindata_1[i])):
            if (traindata_1[i][j]['section_id'] == '1'):
                data_1.append(traindata_1[i][j])
            if (traindata_1[i][j]['section_id'] == '2'):
                data_2.append(traindata_1[i][j])
            if (traindata_1[i][j]['section_id'] == '3'):
                data_3.append(traindata_1[i][j])
            if (traindata_1[i][j]['section_id'] == '4'):
                data_4.append(traindata_1[i][j])
            if (traindata_1[i][j]['section_id'] == '5'):
                data_5.append(traindata_1[i][j])

    data = []
    data.append(data_1)
    data.append(data_2)
    data.append(data_3)
    data.append(data_4)
    data.append(data_5)
    return data

Train_data = Train_data(train_org_data_map)
file = open('Train_data_no_merge.pickle', 'wb')
pickle.dump(Train_data, file)
