#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Tue Jan  9 20:32:28 2018

@author: shixiaohan
"""

import wave
import numpy as np
import librosa
import os

rootdir = '/home/shixiaohan-toda/Desktop/FastSpeech2-master/output/result'
Out_dir = '/home/shixiaohan-toda/Desktop/HNU_Code-main/TTS_Series/TTS_data/'
def read_file(filename):
    file = wave.open(filename, 'r')
    params = file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    str_data = file.readframes(wav_length)
    wavedata = np.fromstring(str_data, dtype=np.short)
    time = np.arange(0, wav_length) * (1.0 / framerate)
    file.close()
    return wavedata, time, framerate

train_num = 0
for speaker in os.listdir(rootdir):
    if (speaker[0] == 'L'):
        sub_dir = os.path.join(rootdir, speaker)
        for sess in os.listdir(sub_dir):
            file_dir = os.path.join(sub_dir, sess)
            wavname = file_dir.split("/")[-1][:-4]
            audio_48k, sr = librosa.load(file_dir, 22050)
            audio_16k = librosa.resample(y=audio_48k, orig_sr=sr, target_sr=16000)
            new_filename = Out_dir + '/' + wavname + '_' + speaker + '.wav'
            librosa.output.write_wav(new_filename, audio_16k, 16000)
            train_num = train_num + 1
            print(train_num)
