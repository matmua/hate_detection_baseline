from sklearn.model_selection import train_test_split
from  sklearn.model_selection import GridSearchCV
from sklearn import svm
import joblib

import os
import numpy as np
import jieba.analyse





# 直接词向量相加求平均
def fea_sentence(list_w):
    n0 = np.array([0. for i in range(100)], dtype=np.float32)
    for i in list_w:
        n0 += i
    fe = n0 / len(list_w)
    fe = fe.tolist()
    return fe

def parse_dataset(x_data, word2vec):
    xVec = []
    for x in x_data:
        sentence = []
        for word in x:
            if word in word2vec:
                sentence.append(word2vec[word])
            else:  # 词不存在，则补零向量。
                sentence.append([0. for i in range(100)])
        xVec.append(fea_sentence(sentence))

    xVec = np.array(xVec)

    return xVec

def get_data(word2vec, data, y):
    data = parse_dataset(data, word2vec)
    x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=5)

    return x_train, y_train, x_test, y_test



def train_svm(x_train, y_train):
    svc = svm.SVC(verbose=True)
    parameters = {'C': [1, 2], 'gamma': [0.5, 1, 2]}
    clf = GridSearchCV(svc, parameters, scoring='f1')
    clf.fit(x_train, y_train, )
    print('最佳参数: ')
    print(clf.best_params_)

    # clf = svm.SVC(kernel='rbf', C=2, gamma=2, verbose=True)
    # clf.fit(x_train,y_train)

    # 封装模型
    print('保存模型...')
    joblib.dump(clf, 'svm.pkl')


