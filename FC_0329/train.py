import os
import time
import random
import argparse
import pickle
import copy
import torch
import numpy as np
import torch.utils.data as Data
import torch.nn.utils.rnn as rmm_utils
import torch.utils.data.dataset as Dataset
import torch.optim as optim
from utils import Get_data
from torch.autograd import Variable
from models import Utterance_net
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.enabled = False

with open('isGeMAPs_iemocap9946.pickle', 'rb') as file:
    data = pickle.load(file)
     
parser = argparse.ArgumentParser(description="RNN_Model")
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--bid_flag', action='store_false')
parser.add_argument('--batch_first', action='store_false')
parser.add_argument('--batch_size', type=int, default=64,metavar='N')
parser.add_argument('--log_interval', type=int, default=10,metavar='N')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--hidden_layer', type=int, default=256)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--out_class', type=int, default=4)
parser.add_argument('--utt_insize', type=int, default=88)
args = parser.parse_args()
'''
训练参数：
--cuda: 使用GPU
--batch_size：training batch 
--dropout：
--epochs： training times
GRU参数：
--bid_flag: 
--batch_first:
Padding:
--utt_insize : 务必与谱信息的dim对应。
'''
torch.manual_seed(args.seed)

def Train(epoch):
    train_loss = 0
    utt_net.train()
    for batch_idx,(data_1,target) in enumerate(train_loader):
        if args.cuda:
            data_1,target = data_1.cuda(),target.cuda()
        data_1, target = Variable(data_1),Variable(target)
        target = target.squeeze()
        utt_optim.zero_grad()
        data_1 = data_1.squeeze()
        utt_out = utt_net(data_1)
        loss = torch.nn.CrossEntropyLoss()(utt_out, target.long())

        loss.backward()

        utt_optim.step()
        train_loss += loss

        if batch_idx> 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                100.* batch_idx / len(train_loader), train_loss.item() / args.log_interval
            ))
            train_loss = 0

def Test():
    utt_net.eval()

    label_pre = []
    label_true = []
    with torch.no_grad():
        for batch_idx,(data_1,target) in enumerate(test_loader):
            if args.cuda:
                data_1,target = data_1.cuda(),target.cuda()
            data_1, target = Variable(data_1),Variable(target)
            utt_optim.zero_grad()
            data_1 = data_1.squeeze()
            utt_out = utt_net(data_1)
            output = torch.argmax(utt_out, dim=1)
            label_true.extend(target.cpu().data.numpy())
            label_pre.extend(output.cpu().data.numpy())
        #print(label_true)
        #print(label_pre)
        accuracy_recall = recall_score(label_true, label_pre, average='macro')
        accuracy_f1 = metrics.f1_score(label_true, label_pre, average='macro')
        CM_test = confusion_matrix(label_true,label_pre)
        print(accuracy_recall)
        print(accuracy_f1)
        print(CM_test)
    return accuracy_f1, accuracy_recall,label_pre,label_true


Final_result = []
Fineal_f1 = []
kf = KFold(n_splits=10)
for index,(train, test) in enumerate(kf.split(data)):
    print(index)
    train_loader,test_loader, input_test_data_id, input_test_label_org = Get_data(data,train,test,args)
    utt_net = Utterance_net(args.utt_insize, args.hidden_layer, args.out_class, args)

    if args.cuda:
        utt_net = utt_net.cuda()
        
    lr = args.lr
    utt_optimizer = getattr(optim, args.optim)(utt_net.parameters(), lr=lr)
    utt_optim = optim.Adam(utt_net.parameters(), lr=lr)
    f1 = 0
    recall = 0
    for epoch in range(1,args.epochs + 1):
        Train(epoch)
        accuracy_f1, accuracy_recall,pre_label,true_label = Test()
        if epoch % 15 == 0:
            lr /= 10
            for param_group in utt_optimizer.param_groups:
                param_group['lr'] = lr

        if(accuracy_f1>f1 and accuracy_recall>recall):
            name_1 = 'utt_net'+str(index)+'.pkl'
            torch.save(utt_net.state_dict(), name_1)
            recall = accuracy_recall
            f1 = accuracy_f1