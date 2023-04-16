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


