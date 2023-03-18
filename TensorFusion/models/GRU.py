import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 04:00:26 2019

@author: SHI Xiaohan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Utterance_net(nn.Module):
    def __init__(self, input_size, args):
        super(Utterance_net, self).__init__()
        self.input_size = input_size
        self.hidden_dim = args.hidden_layer
        self.out_class = args.out_class
        self.drop = nn.Dropout(0.2)
        self.norm = nn.BatchNorm1d(input_size)
        self.input2hidden = nn.Linear(input_size, self.hidden_dim * 2)
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.out_class)

    def forward(self, input):
        # linear
        normed = self.norm(input)
        dropped = self.drop(normed)
        x = F.relu(self.input2hidden(dropped))
        #y = self.hidden2label(x)
        return x



class Utterance_net_out(nn.Module):
    def __init__(self, input_size, args):
        super(Utterance_net_out, self).__init__()
        self.input_size = input_size
        self.hidden_dim = args.hidden_layer
        self.out_class = args.out_class
        self.drop = nn.Dropout(0.2)
        self.norm = nn.BatchNorm1d(input_size)
        self.input2hidden = nn.Linear(input_size, self.hidden_dim * 2)
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.out_class)

    def forward(self, input):
        # linear
        normed = self.norm(input)
        dropped = self.drop(normed)
        x = self.input2hidden(dropped)
        y = self.hidden2label(x)
        return y

