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
import pickle
import os

with open('Speech_data.pickle', 'rb') as file:
    Speech_data = pickle.load(file)

rootdir = '/home/shixiaohan-toda/Desktop/HNU_Code-main/TTS_Series/TTS_data'
#list_id = ['LibriTTS_329', 'LibriTTS_492', 'LibriTTS_1079', 'LibriTTS_1958', 'LibriTTS_2393','LibriTTS_2999','LibriTTS_3307','LibriTTS_6099','LibriTTS_6497','LibriTTS_6895']
list_id = ['LibriTTS_329', 'LibriTTS_492', 'LibriTTS_1079', 'LibriTTS_1958', 'LibriTTS_2393']

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
        #if (speaker[0] == 'L'):
        if (speaker in list_id):
            sub_dir = os.path.join(rootdir, speaker)
            for sess in os.listdir(sub_dir):
                wav_dir = os.path.join(sub_dir, sess)
                wavname = wav_dir.split("/")[-1][:-4]
                data, time, rate = read_file(wav_dir)
                mel_spec = ps.logfbank(data, rate, nfilt=filter_num)
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

def normalization(data,name):
    need_norm = []
    for i in range(len(data)):
        need_norm.append(data[i][name][0])
    a = []
    for i in range(len(need_norm)):
        a.extend(need_norm[i])
    Scaler = StandardScaler().fit(a)
    print(Scaler.mean_)
    print(Scaler.var_)
    for i in range(len(data)):
        data[i][name][0] = Scaler.transform(data[i][name][0])
    return data,Scaler
def Compare_id(Speech_data,data):
    id = []
    for i in range(len(Speech_data)):
        for j in range(len(Speech_data[i])):
            id.append(Speech_data[i][j]['id'])
    all_data = []
    for i in range(len(data)):
        if (data[i]['id'] in id):
            all_data.append(data[i])
    print(len(all_data))
    return all_data
if __name__ == '__main__':
    train_data_spec = Read_IEMOCAP_Spec()
    Fin_data = Compare_id(Speech_data,train_data_spec)
    _ , Scaler = normalization(Fin_data,'spec_data')
    file = open('Scaler.pickle', 'wb')
    pickle.dump(Scaler, file)
    file.close()