import os
import argparse
import pickle
import copy
import torch
import torch.optim as optim
from utils import Get_data
from torch.autograd import Variable
from models import Utterance_net,Utterance_net_out
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from torch.nn.parameter import Parameter

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.enabled = False

with open('Train_data_no_merge.pickle', 'rb') as file:
    data = pickle.load(file)

parser = argparse.ArgumentParser(description="RNN_Model")
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--bid_flag', action='store_false')
parser.add_argument('--batch_first', action='store_false')
parser.add_argument('--batch_size', type=int, default=64, metavar='N')
parser.add_argument('--log_interval', type=int, default=10, metavar='N')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--dia_layers', type=int, default=2)
parser.add_argument('--hidden_layer', type=int, default=256)
parser.add_argument('--out_class', type=int, default=4)
parser.add_argument('--audio_insize', type=int, default=88)
parser.add_argument('--text_insize', type=int, default=768)
args = parser.parse_args()

torch.manual_seed(args.seed)


def Train(epoch):
    train_loss = 0
    dia_net_a.train()
    dia_net_b.train()
    dia_net_all.train()
    # label: emotion_label
    for batch_idx, (data_1, data_2, target) in enumerate(train_loader):
        if args.cuda:
            data_1, data_2, target = data_1.cuda(), data_2.cuda(),target.cuda()
        data_1, data_2, target= Variable(data_1), Variable(data_2),Variable(target)

        target = target.squeeze()

        data_1 = data_1.squeeze()
        data_2 = data_2.squeeze()

        dia_out_a = dia_net_a(data_1)
        dia_out_b = dia_net_b(data_2)

        '''
        ### concat
        #dia_int_all = torch.cat((dia_out_a,dia_out_b), 1)
        '''

        ### TFN
        n = dia_out_a.shape[0]
        # 用 1 扩充维度
        A = torch.cat([dia_out_a, torch.ones(n, 1).cuda()], dim=1)
        B = torch.cat([dia_out_b, torch.ones(n, 1).cuda()], dim=1)
        # 计算笛卡尔积
        A = A.unsqueeze(2)  # [n, A, 1]
        B = B.unsqueeze(1)  # [n, 1, B]
        fusion_AB = torch.einsum('nxt, nty->nxy', A, B)  # [n, A, B]
        dia_int_all = fusion_AB.flatten(start_dim=1)  # [n, AxB, 1]

        '''
        ### LWF
        n = dia_out_a.shape[0]
        A = torch.cat([dia_out_a, torch.ones(n, 1).cuda()], dim=1)
        B = torch.cat([dia_out_b, torch.ones(n, 1).cuda()], dim=1)

        # 假设所设秩: R = 4, 期望融合后的特征维度: h = 128
        R, h = 128, 512
        Wa = Parameter(torch.Tensor(R, A.shape[1], h)).cuda()
        Wb = Parameter(torch.Tensor(R, B.shape[1], h)).cuda()
        Wf = Parameter(torch.Tensor(1, R)).cuda()
        bias = Parameter(torch.Tensor(1, h)).cuda()

        # 分解后，并行提取各模态特征
        fusion_A = torch.matmul(A, Wa)
        fusion_B = torch.matmul(B, Wb)

        # 利用一个Linear再进行特征融合（融合R维度）
        funsion_ABC = fusion_A * fusion_B
        dia_int_all = torch.matmul(Wf, funsion_ABC.permute(1, 0, 2)).squeeze() + bias
        '''
        dia_out_all = dia_net_all(dia_int_all)

        dia_net_a_optimizer.zero_grad()
        dia_net_b_optimizer.zero_grad()
        dia_net_all_optimizer.zero_grad()

        loss = torch.nn.CrossEntropyLoss()(dia_out_all, target.long())

        loss.backward()

        dia_net_a_optimizer.step()
        dia_net_b_optimizer.step()
        dia_net_all_optimizer.step()

        train_loss += loss

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), train_loss.item() / args.log_interval
            ))
            train_loss = 0
def Test():
    dia_net_a.eval()
    dia_net_b.eval()
    dia_net_all.eval()


    label_pre = []
    label_true = []


    with torch.no_grad():
        for batch_idx, (data_1, data_2, target) in enumerate(
                test_loader):

            if args.cuda:
                data_1, data_2, target = data_1.cuda(), data_2.cuda(), target.cuda()
            # data (batch_size, step, 88)
            # target (batch_size, 1)
            data_1, data_2, target = Variable(data_1), Variable(data_2), Variable(target)

            target = target.squeeze()

            data_1 = data_1.squeeze()
            data_2 = data_2.squeeze()

            dia_out_a = dia_net_a(data_1)
            dia_out_b = dia_net_b(data_2)

            '''
            ### concat
            #dia_int_all = torch.cat((dia_out_a,dia_out_b), 1)
            '''

            ### TFN
            n = dia_out_a.shape[0]
            # 用 1 扩充维度
            A = torch.cat([dia_out_a, torch.ones(n, 1).cuda()], dim=1)
            B = torch.cat([dia_out_b, torch.ones(n, 1).cuda()], dim=1)
            # 计算笛卡尔积
            A = A.unsqueeze(2)  # [n, A, 1]
            B = B.unsqueeze(1)  # [n, 1, B]
            fusion_AB = torch.einsum('nxt, nty->nxy', A, B)  # [n, A, B]
            dia_int_all = fusion_AB.flatten(start_dim=1)  # [n, AxB, 1]

            '''
            ### LWF
            n = dia_out_a.shape[0]
            A = torch.cat([dia_out_a, torch.ones(n, 1).cuda()], dim=1)
            B = torch.cat([dia_out_b, torch.ones(n, 1).cuda()], dim=1)

            # 假设所设秩: R = 4, 期望融合后的特征维度: h = 128
            R, h = 128, 512
            Wa = Parameter(torch.Tensor(R, A.shape[1], h)).cuda()
            Wb = Parameter(torch.Tensor(R, B.shape[1], h)).cuda()
            Wf = Parameter(torch.Tensor(1, R)).cuda()
            bias = Parameter(torch.Tensor(1, h)).cuda()

            # 分解后，并行提取各模态特征
            fusion_A = torch.matmul(A, Wa)
            fusion_B = torch.matmul(B, Wb)

            # 利用一个Linear再进行特征融合（融合R维度）
            funsion_ABC = fusion_A * fusion_B
            dia_int_all = torch.matmul(Wf, funsion_ABC.permute(1, 0, 2)).squeeze() + bias
            '''
            dia_out_all = dia_net_all(dia_int_all)

            output = torch.argmax(dia_out_all, dim=1)
            label_true.extend(target.cpu().data.numpy())
            label_pre.extend(output.cpu().data.numpy())


        accuracy_recall = recall_score(label_true, label_pre, average='macro')
        accuracy_f1 = metrics.f1_score(label_true, label_pre, average='macro')
        CM_test = confusion_matrix(label_true, label_pre)


        print("########################################")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(accuracy_recall)
        print(accuracy_f1)
        print(CM_test)
        print("########################################")
    return accuracy_f1, accuracy_recall, label_pre, label_true


Final_result = []
Final_f1 = []
kf = KFold(n_splits=5)
for index, (train, test) in enumerate(kf.split(data)):
    print(index)

    train_loader, test_loader,input_test_data_id, input_test_label_org, test_len = Get_data(data, train, test, args)

    dia_net_a = Utterance_net(args.audio_insize, args)
    dia_net_b = Utterance_net(args.text_insize, args)
    dia_net_all = Utterance_net_out(263169 , args)

    if args.cuda:
        dia_net_a = dia_net_a.cuda()
        dia_net_b = dia_net_b.cuda()
        dia_net_all = dia_net_all.cuda()


    lr = args.lr
    dia_net_a_optimizer = getattr(optim, args.optim)(dia_net_a.parameters(), lr=lr)
    dia_net_b_optimizer = getattr(optim, args.optim)(dia_net_b.parameters(), lr=lr)
    dia_net_all_optimizer = getattr(optim, args.optim)(dia_net_all.parameters(), lr=lr)


    dia_net_a_optim = optim.Adam(dia_net_a.parameters(), lr=lr)
    dia_net_b_optim = optim.Adam(dia_net_b.parameters(), lr=lr)
    dia_net_all_optim = optim.Adam(dia_net_all.parameters(), lr=lr)

    f1 = 0
    recall = 0
    result_label = []
    for epoch in range(1, args.epochs + 1):
        Train(epoch)
        accuracy_f1, accuracy_recall, pre_label, true_label = Test()
        if epoch % 10 == 0:
            lr /= 10
            for param_group in dia_net_a_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in dia_net_b_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in dia_net_all_optimizer.param_groups:
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

file = open('Final_result.pickle', 'wb')
pickle.dump(Final_result, file)
file.close()
file = open('Final_f1.pickle', 'wb')
pickle.dump(Final_f1, file)
file.close()

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