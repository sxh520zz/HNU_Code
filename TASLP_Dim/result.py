#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:10:17 2019

@author: jdang03
"""

import pickle
import numpy as np 
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import mean_squared_error


with open('Final_result.pickle', 'rb') as file:
    final_result =pickle.load(file)
with open('Final_f1.pickle', 'rb') as file:
    Final_f1 =pickle.load(file)

def concordance_correlation_coefficient(y_true,Y_pred,
                                        sample_weight=None,
                                        multioutput='uniform_average'):
    """Concordance correlation coefficient.
    The concordance correlation coefficient is a measure of inter-rater agreement.
    It measures the deviation of the relationship between predicted and true values
    from the 45 degree angle.
    Read more: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation coefficient to evaluate reproducibility." Biometrics (1989): 255-268.
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
    between the true and the predicted values.
    Examples
    --------
    0.97678916827853024
    """

    y_pred = []
    for i in range(len(Y_pred)):
        y_pred.append(Y_pred[i][0])

    cor = np.corrcoef(y_true, y_pred)[0][1]

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator


print(Final_f1)
'''
cret = 0    
num = 0   
for i in range(len(final_result)):
    for j in range(len(final_result[i])):
        for x in range(len(final_result[i][j]['predict_label'])):
            num = num + 1
            if(final_result[i][j]['predict_label'][x] == final_result[i][j]['true_label'][x]):
                cret  = cret + 1
print(cret,num,cret/num)
'''                
true_label = []    
predict_label = []   
num = 0
for i in range(len(final_result)):
    for j in range(len(final_result[i])):
        predict_label.append(final_result[i][j]['Predict_label'])
        true_label.append(final_result[i][j]['True_label'])
print(num)
'''
accuracy_recall = recall_score(true_label, predict_label, average='macro')
accuracy_f1 = metrics.f1_score(true_label, predict_label, average='macro')
CM_test = confusion_matrix(true_label,predict_label)    
'''
accuracy_recall = concordance_correlation_coefficient(true_label, predict_label)
accuracy_f1 = mean_squared_error(true_label, predict_label)


           
print(accuracy_recall,accuracy_f1)