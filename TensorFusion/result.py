#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:10:17 2019

@author: jdang03
"""

import pickle
import numpy as np 
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix 

with open('Final_result.pickle', 'rb') as file:
    final_result =pickle.load(file)
with open('Final_f1.pickle', 'rb') as file:
    Final_f1 =pickle.load(file)

print(Final_f1)
true_label = []
predict_label = []
for i in range(len(final_result)):
    for j in range(len(final_result[i])):
        predict_label.append(final_result[i][j]['Predict_label'])
        true_label.append(final_result[i][j]['True_label'])

accuracy_recall = recall_score(true_label, predict_label, average='macro')
accuracy_acc = accuracy_score(true_label, predict_label)
CM_test = confusion_matrix(true_label, predict_label)

print(len(true_label))
print(accuracy_recall, accuracy_acc)
print(CM_test)