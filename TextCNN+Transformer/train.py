import json

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm
import DataSet
import numpy as np
from sklearn.metrics import accuracy_score,recall_score, f1_score, precision_score
from sklearn.metrics import classification_report
from Config import *
from model.TextCNN import TextCNN
from model.Transformer import Transformer

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

criterion = torch.nn.CrossEntropyLoss()
commentf = './programming.txt'
result_f = './test.txt'

#初始化模型
name = 'TextCNN'
model = TextCNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
train_iter, val_iter, test_iter = DataSet.getIter()

acc_list = []
loss_list = []
precision_list = []
recall_list = []
f1_list = []

def test_model(test_iter, name, device,num_n):#加个num_n
    print('testing...')
    model = torch.load('done_model/'+name+'_model.pkl')
    print('模型加载完成！！！')
    model = model.to(device)
    model.eval()
    for epoch in range(1):
        acc = []
        loss_list = []
        # total_loss = 0.0
        # accuracy = 0
        y_true = []
        y_pred = []
        acc_all = 0
        loss_all = 0
        itercount=0
        total_test_num = len(test_iter.dataset)

        for batch in test_iter:
            itercount=itercount+1
            feature = batch.text
            target = batch.label
            with torch.no_grad():
                feature = torch.t(feature)
            feature, target = feature.to(device), target.to(device)
            out = model(feature)
            print(out)
            count = 0
            with open(commentf, 'a', encoding='utf-8') as f:
                for i in out:
                    f.write('{} {} {}\n'.format(target[count], i[0], i[1]))
                    count = count + 1
            loss = F.cross_entropy(out, target)
            loss_list.append(float(loss))
            #total_loss += loss.item()
            #accuracy += (torch.argmax(out, dim=1)==target).sum().item()
            accuracy = (torch.argmax(out, dim=1) == target).sum().item()
            acc.append(accuracy)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())

        print(itercount)
        for num in range(len(acc)):
            acc_all = acc_all + acc[num]
            loss_all = loss_all + loss_list[num]
        acc_all = acc_all / len(acc)
        loss_all = loss_all / len(loss_list)
        print(loss_all, acc_all)
        measure_result = classification_report(y_true, y_pred)
        print('measure_result = \n', measure_result)
        print("accuracy：%.2f" % accuracy_score(y_true, y_pred))
        print("precision：%.2f" % precision_score(y_true, y_pred))
        print("recall：%.2f" % recall_score(y_true, y_pred))
        print("f1-score：%.2f" % f1_score(y_true, y_pred))
        acc_list.append(accuracy_score(y_true, y_pred))
        loss_list.append(loss_all)
        precision_list.append(precision_score(y_true, y_pred))
        recall_list.append(recall_score(y_true, y_pred))
        f1_list.append(f1_score(y_true, y_pred))
        with open(result_f, 'a', encoding='utf-8') as f:
            f.write(json.dumps(num_n, ensure_ascii=False))
            f.write(",\n")
            f.write(json.dumps("accuracy：%.2f" % accuracy_score(y_true, y_pred), ensure_ascii=False))
            f.write(",\n")
            f.write(json.dumps("precision：%.2f" % precision_score(y_true, y_pred),
                                   ensure_ascii=False))
            f.write(",\n")
            f.write(json.dumps("recall：%.2f" % recall_score(y_true, y_pred), ensure_ascii=False))
            f.write(",\n")
            f.write(json.dumps("f1-score：%.2f" % f1_score(y_true, y_pred), ensure_ascii=False))
            f.write(",\n")
        break

    # print('>>> Test loss:{}, Accuracy:{} \n'.format(total_loss/total_test_num, accuracy/total_test_num))
    # score = accuracy_score(y_true, y_pred)
    # print(score)
    # print(classification_report(y_true, y_pred, digits=3))
    print('test_end')

def train_model(train_iter, dev_iter, model, name, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.6)
    model.train()
    best_acc = 0

    #局部参数
    acc = []
    loss_list = []
    label_true = []
    label_pred = []
    acc_all = 0
    loss_all = 0

    print('training...')
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        accuracy = 0
        total_train_num = len(train_iter.dataset)
        progress_bar = tqdm(enumerate(train_iter), total=len(train_iter))
        for i,batch in progress_bar:
            feature = batch.text
            target = batch.label
            with torch.no_grad():
                feature = torch.t(feature)
            feature, target = feature.to(device), target.to(device)
            optimizer.zero_grad()
            logit = model(feature)#预测结果
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            accuracy += (torch.argmax(logit, dim=1) == target).sum().item()
            progress_bar.set_description(
            f'loss: {loss.item():.3f}')
        print('>>> Epoch_{}, Train loss is {}, Accuracy:{} \n'.format(epoch,loss.item()/total_train_num, accuracy/total_train_num))
        model.eval()
        total_loss = 0.0
        accuracy = 0
        total_valid_num = len(dev_iter.dataset)
        progress_bar = tqdm(enumerate(dev_iter), total=len(dev_iter))
        for i, batch in progress_bar:
            feature = batch.text  # (W,N) (N)
            target = batch.label
            with torch.no_grad():
                feature = torch.t(feature)
            feature, target = feature.to(device), target.to(device)
            out = model(feature)
            loss = F.cross_entropy(out, target)
            total_loss += loss.item()
            accuracy += (torch.argmax(out, dim=1)==target).sum().item()
        print('>>> Epoch_{}, Valid loss:{}, Accuracy:{} \n'.format(epoch, total_loss/total_valid_num, accuracy/total_valid_num))

        if(accuracy/total_valid_num > best_acc):
            print('save model...')
            best_acc = accuracy/total_valid_num
            saveModel(model, name=name)
            print('\n')

def saveModel(model,name):
    torch.save(model, 'done_model/'+name+'_model.pkl')



if __name__ == '__main__':
    train_model(train_iter, val_iter, model, name, device)
    test_model(test_iter, name, device,25)

    # for process in range(10):
    #     print('progress:'+str(process))
    #     data_path = "data/N折交叉验证/ndata_" + str(process + 1)+"/"
    #     train_model(train_iter, val_iter, model, name, device)
    #     test_model(test_iter, name, device,process + 1)
    #
    # acc_total = 0
    # loss_total = 0
    # pre_total = 0
    # recall_total = 0
    # f1_total = 0
    # for num in range(len(acc_list)):
    #     acc_total += acc_list[num]
    #     loss_total += loss_list[num]
    #     pre_total += precision_list[num]
    #     recall_total += recall_list[num]
    #     f1_total += f1_list[num]
    # print("endExperiment:")
    # print("acc:" + str(acc_total / 10) + " | " +
    #       "loss:" + str(loss_total / 10) + " | " +
    #       "precision:" + str(pre_total / 10) + " | " +
    #       "recall:" + str(recall_total / 10) + " | " +
    #       "f1:" + str(f1_total / 10))