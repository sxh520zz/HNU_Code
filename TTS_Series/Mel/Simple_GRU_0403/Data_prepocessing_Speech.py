#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Tue Jan  9 20:32:28 2018

@author: shixiaohan
"""
import re
import wave
import numpy as np
import python_speech_features as ps
from sklearn.preprocessing import StandardScaler
import os
import glob
import pickle
import csv

with open('../Scaler.pickle', 'rb') as file:
    Scaler = pickle.load(file)

Data_dir = '/media/shixiaohan-toda/70e9f22b-13a4-429e-acad-a8786e1f1143/DataBase/journal_Data'
rootdir = '/media/shixiaohan-toda/70e9f22b-13a4-429e-acad-a8786e1f1143/DataBase/IEMOCAP_full_release/'
#Smile_data_dir = Data_dir + '/OpenSmile/Ge'

label_list = [1, 2, 3, 4, 5]

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

def read_file(filename):
    file = wave.open(filename, 'r')
    params = file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    str_data = file.readframes(wav_length)
    wavedata = np.fromstring(str_data, dtype=np.short)
    time = np.arange(0, wav_length) * (1.0 / framerate)
    file.close()
    return wavedata, time, framerate

def Read_IEMOCAP_Spec():
    filter_num = 40
    train_num = 0
    train_mel_data = []
    for speaker in os.listdir(rootdir):
        if (speaker[0] == 'S'):
            sub_dir = os.path.join(rootdir, speaker, 'sentences/wav')
            for sess in os.listdir(sub_dir):
                if (sess[7] in ['i','s']):
                    file_dir = os.path.join(sub_dir, sess, '*.wav')
                    files = glob.glob(file_dir)
                    for filename in files:
                        wavname = filename.split("/")[-1][:-4]
                        data, time, rate = read_file(filename)
                        mel_spec = ps.logfbank(data, rate, nfilt=filter_num)
                        if (speaker in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']):
                            # training set
                            mel_data = []
                            one_mel_data = {}
                            part = mel_spec
                            delta1 = ps.delta(mel_spec, 2)
                            delta2 = ps.delta(delta1, 2)
                            input_data_1 = np.concatenate((part, delta1), axis=1)
                            input_data = np.concatenate((input_data_1, delta2), axis=1)
                            mel_data.append(input_data)
                            one_mel_data['id'] = wavname
                            mel_data = np.array(mel_data)
                            one_mel_data['spec_data'] = mel_data
                            train_mel_data.append(one_mel_data)
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
                if (sess[7] in ['i','s']):
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
    print(train_num)

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
    print(train_num_1)

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


def Seg_IEMOCAP(train_data_spec,train_data_text):
    for i in range(len(train_data_text)):
        for x in range(len(train_data_text[i])):
            for y in range(len(train_data_spec)):
                if (train_data_text[i][x]['id'] == train_data_spec[y]['id']):
                    train_data_text[i][x]['spec_data'] = train_data_spec[y]['spec_data']
    num = 0
    train_data_map = []
    for i in range(len(train_data_text)):
        data_map_1 = []
        for x in range(len(train_data_text[i])):
            if (len(train_data_text[i][x]) == 7):
                if(train_data_text[i][x]['label'] in label_list):
                    data_map_1.append(train_data_text[i][x])
                    num = num + 1
        train_data_map.append(data_map_1)
    print(num)

    train_data_map = normalization(train_data_map,'spec_data')
    return train_data_map

def normalization(data,name):
    need_norm = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            need_norm.append(data[i][j][name][0])
    a = []
    for i in range(len(need_norm)):
        a.extend(need_norm[i])
    #Scaler_1 = StandardScaler().fit(a)
    #print(Scaler_1.mean_)
    #print(Scaler_1.var_)
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j][name][0] = Scaler.transform(data[i][j][name][0])
    return data

if __name__ == '__main__':
    train_data_spec = Read_IEMOCAP_Spec()
    train_data_text = Read_IEMOCAP_Text()
    train_data_map = Seg_IEMOCAP(train_data_spec,train_data_text)
    file = open('Speech_data.pickle', 'wb')
    pickle.dump(train_data_map, file)
    file.close()
