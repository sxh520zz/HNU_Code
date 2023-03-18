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
from models import Utterance_net, Utterance_net_attention
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.enabled = False

with open('/media/shixiaohan/资料/IEMOCAP/DataBase/traindata_map.pickle', 'rb') as file:
    data = pickle.load(file)

parser = argparse.ArgumentParser(description="RNN_Model")
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--bid_flag', action='store_false')
parser.add_argument('--batch_first', action='store_false')
parser.add_argument('--batch_size', type=int, default=1,metavar='N')
parser.add_argument('--log_interval', type=int, default=10,metavar='N')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--dia_layers', type=int, default=2)
parser.add_argument('--hidden_layer', type=int, default=256)
parser.add_argument('--out_class', type=int, default=4)
parser.add_argument('--utt_insize', type=int, default=80)
args = parser.parse_args()

torch.manual_seed(args.seed)

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
            #data_1 = data_1.squeeze(0)
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


    name_1 = 'utt_net'+str(index)+'.pkl'
    utt_net.load_state_dict(torch.load(name_1))

    if args.cuda:
        utt_net = utt_net.cuda()
    lr = args.lr
    utt_optimizer = getattr(optim, args.optim)(utt_net.parameters(), lr=lr)
    utt_optim = optim.Adam(utt_net.parameters(), lr=lr)

    f1 = 0
    recall = 0
    for epoch in range(1,args.epochs + 1):
        accuracy_f1, accuracy_recall,pre_label,true_label = Test()
        if(accuracy_f1>f1 and accuracy_recall>recall):
            predict = copy.deepcopy(input_test_label_org)
            num = 0
            for x in range(len(predict)):
                predict[x] = pre_label[num]
                num = num + 1
            result_label = predict
            recall = accuracy_recall
    onegroup_result = []
    for i in range(len(input_test_data_id)):
        a = {}
        a['id'] = input_test_data_id[i]
        a['Predict_label'] = pre_label[i]
        a['True_label'] = input_test_label_org[i]
        onegroup_result.append(a)
    Final_result.append(onegroup_result)
    Fineal_f1.append(accuracy_f1)

file = open('Final_result.pickle', 'wb')
pickle.dump(Final_result,file)
file.close()
file = open('Fineal_f1.pickle', 'wb')
pickle.dump(Fineal_f1,file)
file.close()