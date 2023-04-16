import collections
import pickle
import re
import jieba
import pandas as pd
import torch
import torch.nn as nn


def regex_filter(s_line):
    special_regex = re.compile(r"[\s]+")
    en_regex = re.compile(r"[.…{|}#$%&\'()*+,!-_./:~^;<=>?@★●，。]+")
    zn_regex = re.compile(r"[《》、，“”；～？！：（）【】]+")
    s_line = special_regex.sub(r"", s_line)
    s_line = en_regex.sub(r"", s_line)
    s_line = zn_regex.sub(r"", s_line)
    return s_line


def data_prepare(train_path, test_path):
    word_freqs = collections.Counter()  # 词频
    max_len = 0
    train_set = pd.read_csv(train_path)
    train_sentences = train_set["Sentence"]
    for sentence in train_sentences:
        sentence = regex_filter(sentence)
        words = jieba.cut(sentence)
        x = 0
        for word in words:
            word_freqs[word] += 1
            x += 1
        max_len = max(max_len, x)

    test_set = pd.read_csv(test_path)
    test_sentences = test_set["Sentence"]
    for sentence in test_sentences:
        sentence = regex_filter(sentence)
        words = jieba.cut(sentence)
        x = 0
        for word in words:
            word_freqs[word] += 1
            x += 1
        max_len = max(max_len, x)

    # 构建词频字典
    word2index = {x[0]: i + 2 for i, x in enumerate(word_freqs.most_common(len(word_freqs)))}
    word2index["pad"] = 0
    word2index["unk"] = 1
    # 将词频字典写入文件中保存
    with open('model/word_dict.pickle', 'wb') as handle:
        pickle.dump(word2index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return len(word2index) + 2, max_len


def get_data(path, max_features, sentence_maxlen, embedding_size):
    # 加载分词字典
    with open('model/word_dict.pickle', 'rb') as handle:
        word2index = pickle.load(handle)
    features = []
    labels = []

    data_set = pd.read_csv(path)
    data_sentences = data_set["Sentence"]
    data_labels = data_set["Label"]
    for label, sentence in zip(data_labels, data_sentences):
        if label == 0:
            labels.append([1, 0])
        else:
            labels.append([0, 1])

        sentence = regex_filter(sentence.replace(' ', ''))
        words = jieba.cut(sentence)
        seqs = []
        i = 0
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["unk"])
            i += 1
            if i >= sentence_maxlen:
                break
        if i < sentence_maxlen:
            seqs = seqs + [word2index["pad"]] * (sentence_maxlen - len(seqs))
        features.append(seqs)

    features = torch.LongTensor(features)
    labels = torch.FloatTensor(labels)
    dataset = torch.utils.data.TensorDataset(features, labels)
    return dataset


# if __name__ == "__main__":
#     p_len, s_len = data_prepare("data/CrossValidation/ndata_1/test.csv", "data/CrossValidation/ndata_1/test.csv")
#     print(p_len, s_len)
#     get_data("data/CrossValidation/ndata_1/test.csv", p_len, s_len, 16)
