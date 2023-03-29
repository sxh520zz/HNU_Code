import re
import pickle
import os
import operator
import numpy as np
import csv
from sklearn import preprocessing

rootdir = os.path.dirname(os.path.abspath(''))
print(rootdir)
name = 'isGeMAPs_iemocap9946'
data_file = name + '.csv'
data_file_1 = 'IEMOCAP_10039.csv'

def as_num(x):
    y = '{:.10f}'.format(x)
    return float(y)

def STD(input_fea,name):
    #标准化
    data_1 = []
    for i in range(len(input_fea)):
        data_1.append(input_fea[i][name])
    a = []
    for i in range(len(data_1)):
        a.extend(data_1[i])
    #print(len(a[0]))
    print(len(a))
    scaler_1 = preprocessing.StandardScaler().fit(a)
    print(scaler_1.mean_)
    print(scaler_1.var_)
    for i in range(len(input_fea)):
        input_fea[i][name] = scaler_1.transform(input_fea[i][name])
    return input_fea


traindata = []

num = 0
traindata = []
file = open(data_file,'r')
file_content = csv.reader(file)
for row in file_content:
    if(row[0][0] == 'S'):
        x = []
        data = {}
        for i in range(len(row)):
            if(i != 0):
                x.append(as_num(float(row[i])))
            else:
                data['id'] = row[i]
        x = np.array(x)
        x = x.reshape(-1, 1)
        data['smile_data'] = x.T
        traindata.append(data)



id_label = []
file = open(data_file_1,'r')
file_content = csv.reader(file)
#提取标签
for row in file_content:
    if(row[0][0] == 'S'):
        data = {}
        data['id'] = row[0]
        data['label_cat'] = int(row[5])
        data['label_V'] = float(row[2])
        data['label_A'] = float(row[3])
        data['label_D'] = float(row[4])
        data['speaker'] = row[6]
        data['i/s'] = row[7][0]
        id_label.append(data)
print(len(id_label))


label = [1,2,3,4,5]
#对齐标签-说话人-谱信息
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
                    data['smile_data'] = traindata[i]['smile_data']
                    input_fea.append(data)

input_fea = STD(input_fea,'smile_data')

speaker = ['1','2','3','4','5','6','7','8','9','10']
#按照说话人分折

num = 0
data = [[],[],[],[],[],[],[],[],[],[]]
for i in range(len(input_fea)):
    for j in range(len(speaker)):
        if(input_fea[i]['speaker'] == speaker[j]):
            data[j].append(input_fea[i])
            num = num +1
print(num)

file_name = name + '.pickle'
file = open(file_name, 'wb')
pickle.dump(data,file)
file.close()