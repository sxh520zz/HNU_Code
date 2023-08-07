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
import soundfile as sf
import os
import glob
import pickle
import torch
import torchaudio

rootdir = '../IEMOCAP_full_release/'

from transformers import Wav2Vec2Model, Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")    # 用于提取通用特征，768维

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


def process_wav_file(wav_file, time):
    waveform, sample_rate = torchaudio.load(wav_file)
    target_length = time * sample_rate
    # 将WAV文件裁剪为目标长度
    if waveform.size(1) > target_length:
        waveform = waveform[:, :target_length]
    else:
        # 如果WAV文件长度小于目标长度，则使用填充进行扩展
        padding_length = target_length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, padding_length))

    return waveform, sample_rate

def read_file(filename):
    file = wave.open(filename, 'r')
    params = file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    channel = file.getnchannels()
    sampwidth = file.getsampwidth()
    framerate = file.getframerate()
    frames = file.getnframes()
    duration = frames/framerate
    wav_length = 3 * framerate
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
    #print(train_num)
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
    #print(train_num)

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
    #print(train_num_1)

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

def Read_IEMOCAP_Trad():
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
                        audio_input, sample_rate = process_wav_file(filename,3)
                        #audio_input, sample_rate = sf.read(filename)
                        input_values = processor(audio_input, sampling_rate=sample_rate,
                                                 return_tensors="pt").input_values
                        if (speaker in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']):
                            # training set
                            one_mel_data = {}
                            one_mel_data['id'] = wavname
                            one_mel_data['wav_encodings'] = input_values
                            train_mel_data.append(one_mel_data)
                            train_num = train_num + 1
    #print(train_num)
    return train_mel_data

def Seg_IEMOCAP(train_data_spec,train_data_text,train_data_trad):
    for i in range(len(train_data_text)):
        for x in range(len(train_data_text[i])):
            for y in range(len(train_data_spec)):
                if (train_data_text[i][x]['id'] == train_data_spec[y]['id']):
                    train_data_text[i][x]['spec_data'] = train_data_spec[y]['spec_data']

    for i in range(len(train_data_text)):
        for x in range(len(train_data_text[i])):
            for y in range(len(train_data_trad)):
                if (train_data_text[i][x]['id'] == train_data_trad[y]['id']):
                    train_data_text[i][x]['wav_encodings'] = train_data_trad[y]['wav_encodings']
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
    return train_data_map


if __name__ == '__main__':
    train_data_spec = Read_IEMOCAP_Spec()
    train_data_trad = Read_IEMOCAP_Trad()
    train_data_text = Read_IEMOCAP_Text()
    train_data_map = Seg_IEMOCAP(train_data_spec,train_data_text,train_data_trad)
    print(len(train_data_map))
    #print(train_data_map[0][0])
    file = open('Speech_data.pickle', 'wb')
    pickle.dump(train_data_map, file)
    file.close()
