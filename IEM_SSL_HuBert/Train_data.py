# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:58:02 2019

@author: shixiaohan
"""

import pickle
import math
import numpy as np
# reload a file to a variable
with open('Speech_data.pickle', 'rb') as file:
    train_org_data_map = pickle.load(file)

def write(train_org_data_map):
    with open("IEM.txt","w") as f:
        for i in range(len(train_org_data_map)):
            for j in range(len(train_org_data_map[i])):
                f.write(train_org_data_map[i][j]['id'])
                f.write('\t')
                f.write(train_org_data_map[i][j]['transcription'])
                f.write('\n')

def Train_data(train_map):
    train_data_ALL_1 = []
    label_list= [1,2,3,4,5]
    num = 0
    for i in range(len(train_map)):
        train_data = []
        for j in range(len(train_map[i])):
            data = {}
            data['label'] = train_map[i][j]['label']
            data['wav_encodings'] = train_map[i][j]['wav_encodings']
            data['id'] = train_map[i][j]['id']
            if(data['label'] in label_list):
                if(data['label'] == 5):
                    data['label'] = 2
                data['label'] = data['label'] - 1
                train_data.append(data)
                num = num + 1
        train_data_ALL_1.append(train_data)

    print(len(train_data_ALL_1))
    print(len(train_data_ALL_1[0]))
    print(len(train_data_ALL_1[0][0]))
    print(num)

    data_1 = []
    data_2 = []
    data_3 = []
    data_4 = []
    data_5 = []

    for i in range(len(train_data_ALL_1)):
        for j in range(len(train_data_ALL_1[i])):
            if (train_data_ALL_1[i][j]['id'][4] == '1'):
                data_1.append(train_data_ALL_1[i][j])
            if (train_data_ALL_1[i][j]['id'][4] == '2'):
                data_2.append(train_data_ALL_1[i][j])
            if (train_data_ALL_1[i][j]['id'][4]== '3'):
                data_3.append(train_data_ALL_1[i][j])
            if (train_data_ALL_1[i][j]['id'][4] == '4'):
                data_4.append(train_data_ALL_1[i][j])
            if (train_data_ALL_1[i][j]['id'][4] == '5'):
                data_5.append(train_data_ALL_1[i][j])

    data = []
    data.append(data_1)
    data.append(data_2)
    data.append(data_3)
    data.append(data_4)
    data.append(data_5) 
    return data

write(train_org_data_map)
Train_data = Train_data(train_org_data_map)

file = open('Train_data.pickle', 'wb')
pickle.dump(Train_data, file)


