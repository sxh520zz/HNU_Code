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

Data_dir = '/home/shixiaohan-toda/Desktop/FastSpeech2-master/output/result'
Out_dir = '/home/shixiaohan-toda/Desktop/Experiment/IEM_Add_data/Out_csv/'
print(Data_dir)

import soundfile as sf


def Read_IEMOCAP_Trad(data_dir):
    data_id = data_dir.split("/")[-1].split("_")
    name_speaker = '_'.join(data_id[:-2])
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
            data['id'] = name
            data['name_speaker'] = name_speaker
    return data

def normalization(data):
    need_norm = []
    for i in range(len(data)):
        need_norm.extend(data[i])
    Scaler = StandardScaler().fit(need_norm)
    for i in range(len(data)):
        data[i]= Scaler.transform(data[i])
    return Scaler

num = 0
data = []
for speaker in os.listdir(Out_dir):
    sub_dir = os.path.join(Out_dir, speaker)
    data.append(Read_IEMOCAP_Trad(sub_dir))
    num = num +1
    if(num % 100 == 0):
        print(num)

name_speaker = {}
for i in range(len(data)):
    if(data[i]['name_speaker'] not in name_speaker):
        name_speaker[data[i]['name_speaker']] = []

for i in range(len(data)):
    if(data[i]['name_speaker'] in name_speaker):
        name_speaker[data[i]['name_speaker']].append(data[i]['trad_data'])


out_data = []
for itm in name_speaker:
    data = {}
    data['Scaler'] = normalization(name_speaker[itm])
    data['id'] = itm
    out_data.append(data)
print(len(out_data))

file = open('Scaler_proposed_2.pickle', 'wb')
pickle.dump(out_data, file)
file.close()

'''
sta_data,Scaler = normalization(data)
file = open('Scaler.pickle', 'wb')
pickle.dump(Scaler, file)
file.close()
'''

'''
num = 0
for speaker in os.listdir(Data_dir):
    if (speaker[0] == 'L'):
        sub_dir = os.path.join(Data_dir, speaker)
        for sess in os.listdir(sub_dir):
            wav_dir = os.path.join(sub_dir, sess)
            opensmile_2_CSV(wav_dir,speaker)
            num = num +1
            if(num % 100 == 0):
                print(num)
'''


'''
file = open('Speech_data.pickle', 'wb')
pickle.dump(train_data_map, file)
file.close()
'''




