# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:58:02 2019

@author: shixiaohan
"""

import pickle
#import pandas as pd
import csv
import math
import numpy as np
import pandas as pd

# reload a file to a variable

data_id = 'R3_L2_datainput_hidden2label_dict'

data_input_dir = data_id + '.pickle'
data_out_dir = data_id + '.csv'
with open(data_input_dir, 'rb') as file:
    train_org_data_map = pickle.load(file)

def shift_data(org):
    data_ind = []
    data_ind.append(org['id'])
    data_ind.extend(org['value'].cpu().numpy()[0].tolist())
    return data_ind

out_data = []
for i in range(len(train_org_data_map)):
    out_data.append(shift_data(train_org_data_map[i]))

test = pd.DataFrame(data=out_data)
test.to_csv(data_out_dir,index=0)