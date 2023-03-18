# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:58:02 2019

@author: shixiaohan
"""
import gensim
from bert_serving.client import BertClient
import pickle

# reload a file to a variable
with open('Speech_data.pickle', 'rb') as file:
    train_org_data_map = pickle.load(file)

bc = BertClient()

for i in range(len(train_org_data_map)):
    for j in range(len(train_org_data_map[i])):
        a = train_org_data_map[i][j]['transcription']
        line = gensim.utils.simple_preprocess(a)
        b = ' '.join(line)
        if b:
            train_org_data_map[i][j]['transcription'] = b
        else:
            del train_org_data_map[i][j]['transcription']
data_map = []
for i in range(len(train_org_data_map)):
    data_map_1 = []
    for j in range(len(train_org_data_map[i])):
        if (len(train_org_data_map[i][j]) == 8):
            data_map_1.append(train_org_data_map[i][j])
    data_map.append(data_map_1)

x = 0
for i in range(len(data_map)):
    for j in range(len(data_map[i])):
        z = []
        a = data_map[i][j]['transcription']
        z.append(a)
        data_map[i][j]['transcr_data'] = bc.encode(z)
        x = x + 1
        if (x % 100 == 0):
            print(x)
print(x)


file = open('Text_data.pickle', 'wb')
pickle.dump(data_map, file)