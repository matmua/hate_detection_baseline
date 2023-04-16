import torch
import data_process
import model
from transformers import AdamW
import pandas as pd
import numpy as np
import json
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn import svm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

commentf = './programming.txt'
result_f = './test_distil.txt'


acc_list = []
loss_list = []
precision_list = []
recall_list = []
f1_list = []



def train(datadir, num_n):
    print("train:")
    print(num_n+1)
    # 数据加载
    data_f = "train.csv"
    loader = data_process.dataprocess(data_f, datadir)
    for epoch in range(1):
        acc = []
        loss_list = []
        label_true = []
        label_pred = []
        input_train = []
        label_train = []
        input_test = []
        label_test = []
        acc_all = 0
        loss_all = 0
        print("epoch数：" + str(epoch + 1))
        for i, (input_ids, labels) in enumerate(loader):
            # print(input_ids)
            # print(labels)
            if (torch.cuda.is_available()):
                input_ids, labels = input_ids.to(device), labels.to(device)
            input_train = input_train + input_ids.tolist()
            label_train = label_train + labels.tolist()

    print(len(input_train))
    print("开始啦！")
    clf = svm.SVC(kernel='rbf', probability=True, max_iter=-1)
    clf.fit(input_train, label_train)
    print("结束啦！")
    print(clf)
    # 支持向量
    print(clf.support_vectors_)
    # 属于支持向量的点的 index
    print(clf.support_)
    # 在每一个类中有多少个点属于支持向量
    print(clf.n_support_)

    data_f = "test.csv"
    loader_test = data_process.dataprocess(data_f, datadir)
    for i, (input_ids, labels) in enumerate(loader_test):
        # print(input_ids)
        # print(labels)
        if (torch.cuda.is_available()):
            input_ids, labels = input_ids.to(device), labels.to(device)
        input_test = input_test + input_ids.tolist()
        label_test = label_test + labels.tolist()
    ret = clf.predict_proba(input_test)
    ret_2 = clf.predict(input_test)
    measure_result = classification_report(label_test, ret_2)
    print('measure_result = \n', measure_result)
    print("accuracy：%.4f" % accuracy_score(label_test, ret_2))
    print("precision：%.4f" % precision_score(label_test, ret_2))
    print("recall：%.4f" % recall_score(label_test, ret_2))
    print("f1-score：%.4f" % f1_score(label_test, ret_2))
    '''with open(r'biaoshi.txt', 'w') as f:
        for rank in range(len(ret)):
            f.write(str(label_test[rank]) + " " + str(ret[rank][0]) + " " + str(ret[rank][1]) + '\n')
    print(ret)'''
    with open(result_f, 'a', encoding='utf-8') as f:
        f.write(json.dumps(num_n, ensure_ascii=False))
        f.write(",\n")
        f.write(json.dumps("accuracy：%.2f" % accuracy_score(label_test, ret_2), ensure_ascii=False))
        f.write(",\n")
        f.write(json.dumps("precision：%.2f" % precision_score(label_test, ret_2), ensure_ascii=False))
        f.write(",\n")
        f.write(json.dumps("recall：%.2f" % recall_score(label_test, ret_2), ensure_ascii=False))
        f.write(",\n")
        f.write(json.dumps("f1-score：%.2f" % f1_score(label_test, ret_2), ensure_ascii=False))
        f.write(",\n")




for process in range(1, 10):
    datadir = r"D:\资料\python\项目\data\实验_plus\N折交叉验证\ndata_" + str(process+1)
    train(datadir, process+1)
