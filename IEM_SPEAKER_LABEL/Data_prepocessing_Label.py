#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Tue Jan  9 20:32:28 2018

@author: shixiaohan
"""
import re
import os
import pickle
import pandas as pd




Data_dir = '/media/shixiaohan-toda/70e9f22b-13a4-429e-acad-a8786e1f1143/DataBase'
rootdir = Data_dir + '/IEMOCAP_full_release'

def input_to_dic(data,key):
    if (key in data):
        return data[key]
    else:
        return '-1'

def OUT_IEMOCAP_LABEL(traindata_map):
    fin_data = []
    for i in range(len(traindata_map)):
        data = {}
        data['id'] = traindata_map[i]['id']
        data['emotion_v'] = traindata_map[i]['emotion_v']
        data['emotion_a'] = traindata_map[i]['emotion_a']
        data['emotion_d'] = traindata_map[i]['emotion_d']
        data['label'] = traindata_map[i]['label']
        data['Cat_L1'] = input_to_dic(traindata_map[i],'Cat_L1')
        data['Cat_L2'] = input_to_dic(traindata_map[i],'Cat_L2')
        data['Cat_L3'] = input_to_dic(traindata_map[i],'Cat_L3')
        data['Cat_Spk'] = input_to_dic(traindata_map[i],'Cat_Spk')
        data['Dim_L1_val'] = input_to_dic(traindata_map[i],'Dim_L1_val')
        data['Dim_L1_act'] = input_to_dic(traindata_map[i],'Dim_L1_act')
        data['Dim_L1_dim'] = input_to_dic(traindata_map[i], 'Dim_L1_dim')
        data['Dim_L2_val'] = input_to_dic(traindata_map[i],'Dim_L2_val')
        data['Dim_L2_act'] = input_to_dic(traindata_map[i],'Dim_L2_act')
        data['Dim_L2_dim'] = input_to_dic(traindata_map[i], 'Dim_L2_dim')
        data['Dim_Spk_val'] = input_to_dic(traindata_map[i],'Dim_Spk_val')
        data['Dim_Spk_act'] = input_to_dic(traindata_map[i],'Dim_Spk_act')
        data['Dim_Spk_dim'] = input_to_dic(traindata_map[i], 'Dim_Spk_dim')
        fin_data.append(data)
        #print(data['Dim_L1_val'])
    pd.DataFrame(fin_data).to_csv('IEMOCAP_SPEAKER_LABEL.csv')

def Read_IEMOCAP_LABEL():
    traindata_map_2 = []
    train_num_1 = 0
    for speaker in os.listdir(rootdir):
        if (speaker in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']):
            emoevl = os.path.join(rootdir, speaker, 'dialog/EmoEvaluation')
            for sess in os.listdir(emoevl):
                if (sess[-1] in ['t']):
                    emotdir = emoevl + '/' + sess
                    emot_map = {}
                    flag = 0
                    id_cat = 1
                    id_dim = 1
                    with open(emotdir, 'r') as emot_to_read:
                        while True:
                            line = emot_to_read.readline()
                            if not line:
                                break
                            if (line == '\n' and len(emot_map) != 0):
                                a = emot_map.copy()
                                traindata_map_2.append(a)
                                flag = 0
                                id_cat = 1
                                id_dim = 1
                            if (line[0] == '['):
                                t = line.split()
                                emot_map['id'] = t[3]
                                x = t[5] + t[6] + t[7]
                                x = re.split(r'[,[]', x)
                                y = re.split(r'[]]', x[3])
                                emot_map['emotion_v'] = float(x[1])
                                emot_map['emotion_a'] = float(x[2])
                                emot_map['emotion_d'] = float(y[0])
                                emot_map['label'] = t[4]
                                flag = 1
                                continue
                            if(flag) == 1:
                                t = line.split()
                                if (t[0][2] == 'E' and t[0][0] == 'C'):
                                    str_id = 'Cat_L' + str(id_cat)
                                    emot_map[str_id] = t[-2][:-1]
                                    id_cat = id_cat + 1
                                if (t[0][2] == 'F' and t[0][0] == 'C'):
                                    str_id = 'Cat_Spk'
                                    emot_map[str_id] = t[-2][:-1]
                                if (t[0][2] == 'E' and t[0][0] == 'A'):
                                    str_id = 'Dim_L' + str(id_dim) + '_val'
                                    emot_map[str_id] = t[2][:-1]
                                    str_id = 'Dim_L' + str(id_dim) + '_act'
                                    emot_map[str_id] = t[4][:-1]
                                    str_id = 'Dim_L' + str(id_dim) + '_dim'
                                    emot_map[str_id] = t[6][:-1]
                                    id_dim = id_dim + 1
                                if (t[0][2] == 'F' and t[0][0] == 'A'):
                                    str_id = 'Dim_Spk' + '_val'
                                    emot_map[str_id] = t[2][:-1]
                                    str_id = 'Dim_Spk' + '_act'
                                    emot_map[str_id] = t[4][:-1]
                                    str_id = 'Dim_Spk' + '_dim'
                                    emot_map[str_id] = t[6][:-1]
                                    id_dim = id_dim + 1

    return traindata_map_2

if __name__ == '__main__':
    train_data_text = Read_IEMOCAP_LABEL()
    file = open('IEMOCAP_SPEAKER_LABEL.pickle', 'wb')
    pickle.dump(train_data_text, file)
    file.close()
    OUT_IEMOCAP_LABEL(train_data_text)


