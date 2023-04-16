# coding: utf-8
# 用gensim去做word2vec的处理，用sklearn当中的SVM进行建模
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
# from sklearn.externals import joblib
import joblib
import sys
import importlib
import json
from sklearn.linear_model import LogisticRegression

importlib.reload(sys)
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
# sys.setdefaultencoding('utf8')
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from sklearn.metrics import classification_report


#  载入数据，做预处理(分词)，切分训练集与测试集
def load_file_and_preprocessing():
    # neg=pd.read_excel('./data/neg.xls')
    # pos=pd.read_excel('./data/pos.xls')
    cw = lambda x: list(jieba.cut(x))
    x_train = pd.read_csv('data/N折交叉验证/ndata_1/train.csv', sep=',', header=1, usecols=[0])
    y_train = pd.read_csv('data/N折交叉验证/ndata_1/train.csv', sep=',', header=1, usecols=[1]).values.ravel()
    x_test = (pd.read_csv('data/N折交叉验证/ndata_1/test.csv', sep=',', header=1, usecols=[0]))
    y_test = pd.read_csv('data/N折交叉验证/ndata_1/test.csv', sep=',', header=1, usecols=[1]).values.ravel()
    # 新增一列 word ,存放分好词的评论，pos[0]代表表格第一列
    # pos['words'] = pos[0].apply(cw)
    # neg['words'] = neg[0].apply(cw)
    # np.ones(len(pos)) 新建一个长度为len(pos)的数组并初始化元素全为1来标注好评
    # np.concatenate（）连接数组
    # axis=0 向下执行方法 axis=1向右执行方法
    ##y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))),axis=0)
    # train_test_split：从样本中随机的按比例选取train data和testdata
    # 一般形式：train_test_split(train_data,train_target,test_size=0.4, random_state=0)
    # train_data：所要划分的样本特征集
    # train_target：所要划分的样本结果（标注）
    # test_size：样本占比，如果是整数的话就是样本的数量
    # random_state：是随机数的种子。
    # x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'])), y, test_size=0.2)
    np.save('logastic_data/y_train.npy', y_train)
    np.save('logastic_data/y_test.npy', y_test)
    return x_train, x_test


# 对每个句子的所有词向量取均值，来生成一个句子的vector
def build_sentence_vector(text, size, imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v.wv[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


# 计算词向量
def get_train_vecs(x_train, x_test):
    n_dim = 100
    # 初始化模型和词表
    imdb_w2v = Word2Vec(x_train, vector_size=n_dim, min_count=1)
    imdb_w2v_2 = Word2Vec(x_test, vector_size=n_dim, min_count=1)
    # imdb_w2v = Word2Vec(size=300, window=5, min_count=10, workers=12)
    # imdb_w2v.build_vocab(x_train)
    # imdb_w2v.train(x_train,
    #                total_examples=imdb_w2v.corpus_count,
    #                epochs=imdb_w2v.iter)
    train_vecs = np.zeros((len(x_train), n_dim))
    i = 0
    for z in x_train.values.ravel():
        train_vecs[i] = build_sentence_vector(z, n_dim, imdb_w2v)
        i = i + 1
    # train_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_train])
    # train_vecs = scale(train_vecs)
    print("train_vecs")
    print(train_vecs)
    np.save('logastic_data/train_vecs.npy', train_vecs)
    # 在测试集上训练
    imdb_w2v.train(x_test, total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.epochs)
    # imdb_w2v.train(x_test,
    #                total_examples=imdb_w2v.corpus_count,
    #                epochs=imdb_w2v.iter)
    imdb_w2v.save('logastic_data/w2v_model/w2v_model.pkl')
    # Build test tweet vectors then scale
    # test_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_test])
    test_vecs = np.zeros((len(x_test), n_dim))
    j = 0
    for k in x_test.values.ravel():
        test_vecs[j] = build_sentence_vector(k, n_dim, imdb_w2v)
        j = j + 1
    print("test_vecs")
    print(test_vecs)
    # test_vecs = scale(test_vecs)
    np.save('logastic_data/test_vecs.npy', test_vecs)


def get_data():
    train_vecs = np.load('logastic_data/train_vecs.npy')
    y_train = np.load('logastic_data/y_train.npy')
    test_vecs = np.load('logastic_data/test_vecs.npy')
    y_test = np.load('logastic_data/y_test.npy')
    return train_vecs, y_train, test_vecs, y_test


# 训练logastic模型
def logastic_train(train_vecs, y_train, test_vecs, y_test):
    clf= LogisticRegression()
    clf.fit(train_vecs, y_train)
    joblib.dump(clf, 'logastic_data/svm_model/model.pkl')
    # print (clf.score(test_vecs,y_test))
    res=np.ones(len(test_vecs))
    res2 = np.ones((len(test_vecs), 2))
    result1 = np.zeros((2, 1))
    result2=np.zeros((2, 1))
    test_vec=np.zeros((2, 100))
    i=0
    for test_vec[0] in test_vecs:
        result1 = (clf.predict(test_vec))
        result2 = (clf.predict_proba(test_vec))
        res[i] = result1[0]
        res2[i]=result2[0]
        i=i+1
    print(res)

    np.savetxt("F:\python1\loga\data\log_pre.txt", res, fmt='%d', delimiter=' ')
    np.savetxt("F:\python1\loga\data\log_pre2.txt", res2, fmt='%f', delimiter=' ')
    #print("\naccuracy")
    #print(accuracy_score(y_test, res))
    # print("\nprecision")
    #print(precision_score(y_test, res))
    # print("\nrecall")
    #print(recall_score(y_test, res))
    # print("\nf1-score")
    #print(f1_score(y_test, res))


# 构建待预测句子的向量

def get_predict_vecs(words):
    n_dim = 300
    imdb_w2v = Word2Vec.load('logastic_data/w2v_model/w2v_model.pkl')
    # imdb_w2v.train(words)
    train_vecs = build_sentence_vector(words, n_dim, imdb_w2v)
    # print train_vecs.shape
    return train_vecs


# 对单个句子进行情感判断

def logastic_predict(string):
    words = jieba.lcut(string)
    words_vecs = get_predict_vecs(words)
    clf = joblib.load('logastic_data/logastic_model/model.pkl')

    result = clf.predict(words_vecs)

    return result[0]

    # if int(result[0])==1:
    #    print(string+' positive')
    # else:
    #    print(string+' negative')


#
# x_train,x_test = load_file_and_preprocessing()
# get_train_vecs(x_train,x_test)
if __name__ == "__main__":
    x_train, x_test = load_file_and_preprocessing()
    get_train_vecs(x_train, x_test)
    print("train begins")
    train_vecs, y_train, test_vecs, y_test = get_data()
    print("main")
    print(train_vecs.shape)
    print(y_train.shape)
    print(test_vecs.shape)
    print(y_test.shape)

    logastic_train(train_vecs, y_train, test_vecs, y_test)
    print("train end")

##对输入句子情感进行判断
# string1='你真厉害呢'

# string='这手机真棒，从1米高的地方摔下去就坏了'
# svm_predict(string1)


