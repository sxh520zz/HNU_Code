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

Data_dir = '/home/shixiaohan-toda/Desktop/HNU_Code-main/TTS_Series/TTS_data'
Out_dir = '/home/shixiaohan-toda/Desktop/HNU_Code-main/TTS_Series/Opensmile/Out_csv/'
print(Data_dir)

def opensmile_2_CSV(dir,id):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    y = smile.process_file(dir)
    name = dir.split('/')
    target_name = name[-1][:-4] + "_" + id +'.csv'
    target_dir_ind = Out_dir + target_name
    y.to_csv(target_dir_ind)

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





