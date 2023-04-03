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
from sklearn.metrics import accuracy_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.enabled = False

with open('Train_data.pickle', 'rb') as file:
    data = pickle.load(file)
     
parser = argparse.ArgumentParser(description="RNN_Model")
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--bid_flag', action='store_false')
parser.add_argument('--batch_first', action='store_false')
parser.add_argument('--batch_size', type=int, default=64,metavar='N')
parser.add_argument('--log_interval', type=int, default=10,metavar='N')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--dia_layers', type=int, default=2)
parser.add_argument('--hidden_layer', type=int, default=256)
parser.add_argument('--out_class', type=int, default=4)
parser.add_argument('--utt_insize', type=int, default=120)
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
--dia_layers
--out_class
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
            data_1 = data_1.squeeze(0)
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
Final_f1 = []
kf = KFold(n_splits=5)
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
    result_label = []
    for epoch in range(1,args.epochs + 1):
        Train(epoch)
        accuracy_f1, accuracy_recall,pre_label,true_label = Test()
        if epoch % 15 == 0:
            lr /= 10
            for param_group in utt_optimizer.param_groups:
                param_group['lr'] = lr

        if (accuracy_f1 > f1 and accuracy_recall > recall):
            predict = copy.deepcopy(input_test_label_org)
            num = 0
            for x in range(len(predict)):
                predict[x] = pre_label[num]
                num = num + 1
            result_label = predict
            recall = accuracy_recall
            f1 = accuracy_f1

    onegroup_result = []

    for i in range(len(input_test_data_id)):
        a = {}
        a['id'] = input_test_data_id[i]
        a['Predict_label'] = result_label[i]
        a['True_label'] = input_test_label_org[i]
        onegroup_result.append(a)
    Final_result.append(onegroup_result)
    Final_f1.append(f1)

true_label = []
predict_label = []
for i in range(len(Final_result)):
    for j in range(len(Final_result[i])):
        predict_label.append(Final_result[i][j]['Predict_label'])
        true_label.append(Final_result[i][j]['True_label'])

accuracy_recall = recall_score(true_label, predict_label, average='macro')
accuracy_acc = accuracy_score(true_label, predict_label)
CM_test = confusion_matrix(true_label, predict_label)

print(len(true_label))
print(accuracy_recall, accuracy_acc)
print(CM_test)