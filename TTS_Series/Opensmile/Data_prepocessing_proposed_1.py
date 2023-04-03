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

with open('Speech_data.pickle', 'rb') as file:
    Speech_data = pickle.load(file)

Data_dir = '/home/shixiaohan-toda/Desktop/HNU_Code-main/TTS_Series/TTS_data'
Out_dir = '/home/shixiaohan-toda/Desktop/HNU_Code-main/TTS_Series/Opensmile/Out_csv/'
print(Data_dir)

import soundfile as sf

#list_id = ['LibriTTS_329', 'LibriTTS_492', 'LibriTTS_1079', 'LibriTTS_1958', 'LibriTTS_2393','LibriTTS_2999','LibriTTS_3307','LibriTTS_6099','LibriTTS_6497','LibriTTS_6895']
list_id = ['LibriTTS_2999','LibriTTS_3307','LibriTTS_6099','LibriTTS_6497','LibriTTS_6895']

def Read_IEMOCAP_Trad(data_dir):

    data_id = data_dir.split("/")[-1].split("_")
    name_speaker = '_'.join(data_id[:-2])
    speaker = '_'.join(data_id[-2:])[:-4]
    name = '_'.join(data_id)
    file = open(data_dir, 'r')
    file_content = csv.reader(file)
    data = {}
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
            w = np.array(x)
            w = w.reshape(1, -1)
            data['trad_data'] = w
            data['name_speaker'] = name_speaker
            data['id'] = name
            data['speaker'] = speaker
    return data

def normalization(data):
    need_norm = []
    for i in range(len(data)):
        need_norm.extend(data[i])
    Scaler = StandardScaler().fit(need_norm)
    return Scaler

num = 0
data = []
for speaker in os.listdir(Out_dir):
    sub_dir = os.path.join(Out_dir, speaker)
    out = Read_IEMOCAP_Trad(sub_dir)
    if (out['speaker'] in list_id):
        data.append(Read_IEMOCAP_Trad(sub_dir))
        num = num +1
        if(num % 100 == 0):
            print(num)

name_speaker = {}
for i in range(len(data)):
    if(data[i]['name_speaker'] not in name_speaker):
        name_speaker[data[i]['name_speaker']] = []

all_data = {}
label_list = [1, 2, 3, 4, 5]
for i in range(len(Speech_data)):
    for j in range(len(Speech_data[i])):
        if(Speech_data[i][j]['id'] in name_speaker):
            if(Speech_data[i][j]['label'] in label_list):
                all_data[Speech_data[i][j]['id']] = []

fin_data = []
for i in range(len(data)):
    if(data[i]['name_speaker'] in all_data):
        fin_data.append(data[i]['trad_data'])

print(len(fin_data))
Scaler = normalization(fin_data)
file = open('Scaler_proposed_1.pickle', 'wb')
pickle.dump(Scaler, file)
file.close()

