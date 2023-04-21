import re
import pickle
import os
import operator
import numpy as np
import csv
from sklearn import preprocessing
import pandas as pd

rootdir = os.path.dirname(os.path.abspath('..'))
print(rootdir)
Datadir = '/media/shixiaohan-toda/70e9f22b-13a4-429e-acad-a8786e1f1143/朋友的实验/xingfeng_music_unfinish/'

Feature_name = 'proposed_csvIEMOCAP_80'
data_file = Datadir + Feature_name

Lab_name = 'IEMOCAP_10039.csv'
data_file_1 = Datadir + Lab_name

def as_num(x):
    y = '{:.10f}'.format(x)
    return float(y)

def STD(input_fea,name):
    data_1 = []
    for i in range(len(input_fea)):
        data_1.append(input_fea[i][name])
    a = []
    for i in range(len(data_1)):
        a.extend(data_1[i])
    print(len(a[0]))
    print(len(a))
    scaler_1 = preprocessing.StandardScaler().fit(a)
    print(scaler_1.mean_)
    print(scaler_1.var_)
    for i in range(len(input_fea)):
        input_fea[i][name] = scaler_1.transform(input_fea[i][name])
    return input_fea

gram_data_dir = data_file
traindata = []

num = 0

for sess in os.listdir(gram_data_dir):
    data_dir = gram_data_dir + '/' + sess
    data_1 = []
    data = {}
    file = open(data_dir,'r')
    file_content = csv.reader(file)
    for row in file_content:
        x = []
        for i in range(len(row)):
            row[i] = float(row[i])
            b = np.isinf(row[i])
            #print(b)
            if b:
                print(row[i])
            x.append(row[i])
        row = np.array(x)
        data_1.append(row)
    data['id'] = sess[:-4]
    data_1_1 = np.array(data_1[:39])
    data['gram_data'] = data_1_1.T
    num = num + 1
    traindata.append(data)
    print(num)
print(len(traindata))

id_label = []
file = open(data_file_1,'r')
file_content = csv.reader(file)
for row in file_content:
    if(row[0][0] == 'S'):
        data = {}
        data['id'] = row[0]
        data['label_cat_org'] = row[1]
        data['label_cat'] = int(row[5])
        data['label_V'] = float(row[2])
        data['label_A'] = float(row[3])
        data['label_D'] = float(row[4])
        data['speaker'] = row[6]
        data['i/s'] = row[7][0]
        id_label.append(data)
print(len(id_label))

#label = [1,2,3,4,5,6,7,8,9,0]
label = [1,2,3,4,5]
input_fea = []
for i in range(len(traindata)):
    for j in range(len(id_label)):
        if(traindata[i]['id'] == id_label[j]['id']):
            if(id_label[j]['label_cat'] in label):
                if(id_label[j]['i/s'] == 'i' or 's'):
                    if(id_label[j]['label_cat'] == 5):
                        id_label[j]['label_cat'] = 2
                    data = {}
                    data['label_cat'] = id_label[j]['label_cat'] - 1
                    data['label_V'] = id_label[j]['label_V']
                    data['label_A'] = id_label[j]['label_A']
                    data['label_D'] = id_label[j]['label_D']
                    data['speaker'] = id_label[j]['speaker']
                    data['id'] = traindata[i]['id']
                    data['gram_data'] = traindata[i]['gram_data']
                    input_fea.append(data)

input_fea = STD(input_fea,'gram_data')

speaker = ['1','2','3','4','5','6','7','8','9','10']
data = [[],[],[],[],[],[],[],[],[],[]]
for i in range(len(input_fea)):
    for j in range(len(speaker)):
        if(input_fea[i]['speaker'] == speaker[j]):
            data[j].append(input_fea[i])

file_name = Feature_name + '.pickle'
file = open(file_name, 'wb')
pickle.dump(data,file)
file.close()