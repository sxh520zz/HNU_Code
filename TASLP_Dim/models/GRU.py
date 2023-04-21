import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Utterance_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, args):
        super(Utterance_net, self).__init__()
        self.hidden_dim = args.hidden_layer
        self.num_layers = args.dia_layers
        #  dropout
        self.dropout = nn.Dropout(args.dropout)
        # gru
        self.bigru = nn.GRU(input_size, self.hidden_dim,
                            batch_first=True, num_layers=self.num_layers, bidirectional=True)
        # linear
        self.hidden2label = nn.Linear(self.hidden_dim * 2, output_size)


    def forward(self, input):
        input = self.dropout(input)
        # gru
        gru_out, _ = self.bigru(input)
        #gru_out = torch.transpose(gru_out, 0, 1)
        gru_out = torch.transpose(gru_out, 1, 2)
        # pooling
        # gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        #embed = self.dropout(input)
        gru_out = F.tanh(gru_out)
        # linear
        y = self.hidden2label(gru_out)
        return y
'''
class Utterance_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, args):
        super(Utterance_net, self).__init__()
        self.hidden_dim = args.hidden_layer
        self.num_layers = args.dia_layers

        self.bilstm = nn.LSTM(input_size, self.hidden_dim, num_layers =self.num_layers,
                              dropout=args.dropout, batch_first=True, bidirectional=True, bias=False)

        self.hidden2label1 = nn.Linear(self.hidden_dim*2, self.hidden_dim // 2)
        self.hidden2label2 = nn.Linear(self.hidden_dim // 2, output_size)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, input):
        embed = self.dropout(input)
        bilstm_out, _ = self.bilstm(embed)
        #bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        y = self.hidden2label1(bilstm_out)
        y = self.hidden2label2(y)
        return y
 '''   
class Utterance_net_attention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, args):
        super(Utterance_net_attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.CUDA_USE = 1
        self.n_layers = args.dia_layers
        self.dropout = nn.Dropout(args.dropout)
        self.bid_flag = args.bid_flag
        self.batch_size = args.batch_size
        self.GRU_net = nn.LSTM(self.input_size, self.hidden_size, self.n_layers,
                              batch_first=True, bidirectional=self.bid_flag)
        self.proj1 = nn.Linear(hidden_size *2, hidden_size *2)
        self.tanh = nn.Tanh()
        self.u = nn.Parameter(torch.Tensor(2*hidden_size,1))
        self.proj2 = nn.Linear(hidden_size * 2, output_size)
        self.init_params()
        if (self.bid_flag == True):
            bid_mul = 2
        if (self.bid_flag == False):
            bid_mul = 1
        self.h = torch.empty(self.n_layers * bid_mul, self.batch_size, self.hidden_size)
        self.h = nn.init.xavier_normal_(self.h)
        self.h = self.h.cuda()
        
        self.h_1 = torch.empty(self.n_layers * bid_mul, self.batch_size, self.hidden_size)
        self.h_1 = nn.init.xavier_normal_(self.h_1)
        self.h_1 = self.h_1.cuda()
        
        self.relu = nn.LeakyReLU()
        self.f1 = nn.Linear(256,128)
        self.f2 = nn.Linear(128, 32)
        self.out_linear = nn.Linear(32, output_size)

    def forward(self, indata):
        #indata = indata.unsqueeze(0)
        indata = self.dropout(indata)
        output, (hn,hc) = self.GRU_net(indata, (self.h,self.h_1))
        ut = self.tanh(self.proj1(output))
        alpha = torch.softmax(torch.matmul(ut, self.u), dim=1)
        s = torch.sum(alpha*output, dim=1) 
        s = self.dropout(s)
        out = self.f1(s)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.f2(out)
        out = self.relu(out)
        out = self.dropout(out)
        output = self.out_linear(out)
        return output

    def init_params(self):
        nn.init.xavier_uniform_(self.proj1.weight.data)
        nn.init.xavier_uniform_(self.proj2.weight.data)
        nn.init.constant_(self.proj1.bias.data,0.1)
        nn.init.constant_(self.proj2.bias.data, 0.1)
        nn.init.uniform_(self.u, -0.1,0.1)


