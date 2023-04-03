# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:58:02 2019

@author: shixiaohan
"""

import pickle
import re
import csv
import os
import glob
import numpy as np
import scipy.io.wavfile as wav
import opensmile
from sklearn.preprocessing import StandardScaler

with open('Scaler_proposed_1.pickle', 'rb') as file:
    Scaler = pickle.load(file)

Data_dir = '/media/shixiaohan-toda/70e9f22b-13a4-429e-acad-a8786e1f1143/DataBase'
rootdir = Data_dir + '/IEMOCAP_full_release'

out_dir = '/home/shixiaohan-toda/Desktop/HNU_Code-main/TTS_Series/Opensmile/Orgin/'
print(rootdir)

def Read_IEMOCAP_Trad():
    Smile_data_dir = out_dir
    traindata = []
    for sess in os.listdir(Smile_data_dir):
        if (sess[0] in ['S']):
            data_dir = Smile_data_dir + '/' + sess
            data_1 = []
            data = {}
            file = open(data_dir, 'r')
            file_content = csv.reader(file)
            for row in file_content:
                if (row[0] != 'file'):
                    x = []
                    for i in range(3, len(row)):
                        row[i] = float(row[i])
                        b = np.isinf(row[i])
                        # print(b)
                        if b:
                            print(row[i])
                        x.append(row[i])
                    row = np.array(x)
                    data_1.append(row)
            data['id'] = sess[:-4]
            data_1_1 = np.array(data_1)
            data['trad_data'] = data_1_1
            traindata.append(data)
    return traindata

def Read_Trad(data_dir):
    data_1 = []
    file = open(data_dir, 'r')
    file_content = csv.reader(file)
    for row in file_content:
        if(len(row) == 90):
            smile_data = row[1:-1]
            x = []
            for i in range(len(smile_data)):
                smile_data[i] = float(smile_data[i])
                b = np.isinf(smile_data[i])
                if b:
                    print(smile_data[i])
                x.append(smile_data[i])
            smile_data = np.array(x)
            data_1.append(smile_data)
    data_1_1 = np.array(data_1)
    return data_1_1

def opensmile_2_CSV(dir):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    y = smile.process_file(dir)
    print(dir)
    name = dir.split('/')
    print(name[-1])
    target_name = name[-1][:-4] + '.csv'
    target_dir_ind = out_dir + target_name
    y.to_csv(target_dir_ind)
    Trad_data = Read_Trad(target_dir_ind)
    return Trad_data

def emo_change(x):
    if x == 'xxx' or x == 'oth':
        x = 0
    if x == 'neu':
        x = 1
    if x == 'hap':
        x = 2
    if x == 'ang':
        x = 3
    if x == 'sad':
        x = 4
    if x == 'exc':
        x = 5
    if x == 'sur':
        x = 6
    if x == 'fea':
        x = 7
    if x == 'dis':
        x = 8
    if x == 'fru':
        x = 9
    return x

def Read_IEMOCAP_Spec():
    train_num = 0
    train_mel_data = []
    for speaker in os.listdir(rootdir):
        if (speaker[0] == 'S'):
            sub_dir = os.path.join(rootdir, speaker, 'sentences/wav')
            for sess in os.listdir(sub_dir):
                if (sess[7] in ['i', 's']):
                    file_dir = os.path.join(sub_dir, sess, '*.wav')
                    files = glob.glob(file_dir)
                    for filename in files:
                        wavname = filename.split("/")[-1][:-4]
                        if (speaker in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']):
                            # training set
                            one_mel_data = {}
                            one_mel_data['id'] = wavname
                            one_mel_data['id_str'] = filename
                            train_mel_data.append(one_mel_data)
                            #opensmile_2_CSV(filename)
                            train_num = train_num + 1
                        print(train_num)
    return train_mel_data

def Read_IEMOCAP_Text():
    traindata_map_1 = []
    train_num = 0
    for speaker in os.listdir(rootdir):
        if (speaker in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']):
            text_dir = os.path.join(rootdir, speaker, 'dialog/transcriptions')
            for sess in os.listdir(text_dir):
                if (sess[7] in ['i', 's']):
                    data_map1_1 = []
                    textdir = text_dir + '/' + sess
                    text_map = {}
                    with open(textdir, 'r') as text_to_read:
                        while True:
                            line = text_to_read.readline()
                            if not line:
                                break
                            t = line.split()
                            if (t[0][0] in 'S'):
                                str = " ".join(t[2:])
                                text_map['id'] = t[0]
                                text_map['transcription'] = str
                                a = text_map.copy()
                                data_map1_1.append(a)
                    traindata_map_1.append(data_map1_1)
                    train_num = train_num + 1
    # print(train_num)

    traindata_map_2 = []
    train_num_1 = 0
    for speaker in os.listdir(rootdir):
        if (speaker in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']):
            emoevl = os.path.join(rootdir, speaker, 'dialog/EmoEvaluation')
            for sess in os.listdir(emoevl):
                if (sess[-1] in ['t']):
                    data_map2_1 = []
                    emotdir = emoevl + '/' + sess
                    # emotfile = open(emotdir)
                    emot_map = {}
                    with open(emotdir, 'r') as emot_to_read:
                        while True:
                            line = emot_to_read.readline()
                            if not line:
                                break
                            if (line[0] == '['):
                                t = line.split()
                                emot_map['id'] = t[3]
                                x = t[5] + t[6] + t[7]
                                x = re.split(r'[,[]', x)
                                y = re.split(r'[]]', x[3])
                                emot_map['emotion_v'] = float(x[1])
                                emot_map['emotion_a'] = float(x[2])
                                emot_map['emotion_d'] = float(y[0])
                                emot_map['label'] = emo_change(t[4])
                                a = emot_map.copy()
                                data_map2_1.append(a)
                    traindata_map_2.append(data_map2_1)
                    train_num_1 = train_num_1 + 1
    # print(train_num_1)

    for i in range(len(traindata_map_1)):
        for j in range(len(traindata_map_1[i])):
            for x in range(len(traindata_map_2)):
                for y in range(len(traindata_map_2[x])):
                    if (traindata_map_1[i][j]['id'] == traindata_map_2[x][y]['id']):
                        traindata_map_1[i][j]['emotion_v'] = traindata_map_2[x][y]['emotion_v']
                        traindata_map_1[i][j]['emotion_a'] = traindata_map_2[x][y]['emotion_a']
                        traindata_map_1[i][j]['emotion_d'] = traindata_map_2[x][y]['emotion_d']
                        traindata_map_1[i][j]['label'] = traindata_map_2[x][y]['label']
    train_data_map = []
    for i in range(len(traindata_map_1)):
        data_map_1 = []
        for x in range(len(traindata_map_1[i])):
            if (len(traindata_map_1[i][x]) == 6):
                data_map_1.append(traindata_map_1[i][x])
        train_data_map.append(data_map_1)
    return train_data_map

def Seg_IEMOCAP_1(train_data_spec, train_data_text):
    num = 0
    for i in range(len(train_data_text)):
        for x in range(len(train_data_text[i])):
            for y in range(len(train_data_spec)):
                if (train_data_text[i][x]['id'] == train_data_spec[y]['id']):
                    train_data_text[i][x]['id_str'] = train_data_spec[y]['id_str']
                    num = num + 1
    print(num)

    num = 0
    train_data_map = []
    for i in range(len(train_data_text)):
        data_map_1 = []
        for x in range(len(train_data_text[i])):
            if (len(train_data_text[i][x]) == 8):
                train_data_text[i][x]['trad_data'] = opensmile_2_CSV(train_data_text[i][x]['id_str'])
                data_map_1.append(train_data_text[i][x])
                num = num + 1
        train_data_map.append(data_map_1)
    print(num)
    train_data_map = normalization(train_data_map,'trad_data')
    return train_data_map

def Seg_IEMOCAP(train_data_spec,train_data_text,train_data_trad):
    for i in range(len(train_data_text)):
        for x in range(len(train_data_text[i])):
            for y in range(len(train_data_spec)):
                if (train_data_text[i][x]['id'] == train_data_spec[y]['id']):
                    train_data_text[i][x]['id_str'] = train_data_spec[y]['id_str']
                    #train_data_text[i][x]['spec_data'] = train_data_spec[y]['spec_data']

    for i in range(len(train_data_text)):
        for x in range(len(train_data_text[i])):
            for y in range(len(train_data_trad)):
                if (train_data_text[i][x]['id'] == train_data_trad[y]['id']):
                    train_data_text[i][x]['trad_data'] = train_data_trad[y]['trad_data']
    num = 0
    train_data_map = []
    for i in range(len(train_data_text)):
        data_map_1 = []
        for x in range(len(train_data_text[i])):
            if (len(train_data_text[i][x]) == 8):
                data_map_1.append(train_data_text[i][x])
                num = num + 1
        train_data_map.append(data_map_1)
    print(num)
    train_data_map = normalization(train_data_map,'trad_data')
    '''
    for i in range(len(train_data_map)):
        for j in range(len(train_data_map[i])):
            train_data_map[i][j] = normalization(train_data_map[i][j], 'trad_data')
    '''

    return train_data_map

def normalization(data,name):
    need_norm = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            need_norm.append(data[i][j][name][0])
    Scaler_1 = StandardScaler().fit(need_norm)
    print(Scaler_1.mean_)
    print(Scaler_1.var_)
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j][name] = Scaler.transform(data[i][j][name])
    return data

    '''
    for i in range(len(Scaler)):
        if(data['id'] == Scaler[i]['id']):
            data[name] = Scaler[i]['Scaler'].transform(data[name])
            print(data['id'])
    return data
    '''

train_data_spec = Read_IEMOCAP_Spec()
train_data_trad = Read_IEMOCAP_Trad()
train_data_text = Read_IEMOCAP_Text()

#train_data_map = Seg_IEMOCAP_1(train_data_spec, train_data_text)
train_data_map = Seg_IEMOCAP(train_data_spec, train_data_text, train_data_trad)


file = open('Speech_data.pickle', 'wb')
pickle.dump(train_data_map, file)
file.close()



