import csv
import json
import os

import torch.nn as nn
import torch
from prepare import data_prepare, get_data
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from sklearn.metrics import classification_report

torch.manual_seed(2020)


class BiLSTM(nn.Module):
    # vocab_size：word2Vec的长度
    # embedding_size：将每条数据的每一个字词展开的维度
    # hidden_dim：隐藏层特征数量，即h的维度
    # num_layers：lstm层数
    # num_directions：单向、双向
    # num_class：输出的维度
    def __init__(self, vocab_size, embedding_size, hidden_dim, num_layers, num_directions, num_class):
        super(BiLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.num_directions = num_directions

        self.embedding = nn.Embedding(vocab_size, embedding_dim=self.embedding_size)
        self.bilstm = nn.LSTM(input_size=self.embedding_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              batch_first=True,
                              bidirectional=(self.num_directions == 2))
        self.liner = nn.Linear(self.num_layers * self.hidden_size * self.num_directions, num_class)
        self.act_fun = nn.Softmax(dim=1)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        output, (h, c) = self.bilstm(embedded)
        output = torch.cat([h[0, :, :], h[1, :, :]], dim=-1)
        output1 = self.liner(output)
        output = self.act_fun(output1)
        return output, output1


def train(model, epochs, train_loader, test_loader, optimizer, loss_func, result_path):
    for epoch in range(epochs):
        model.train()
        loss_val = 0.0
        corrects = 0.0

        for datas, labels in train_loader:
            datas = datas.to(device)
            labels = labels.to(device)
            preds, _ = model(datas)

            loss = loss_func(preds, labels)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            loss_val += loss.item() * datas.size(0)

            preds = torch.argmax(preds, dim=1)
            labels = torch.argmax(labels, dim=1)
            corrects += torch.sum(preds == labels).item()

        train_loss = loss_val / len(train_loader.dataset)
        train_acc = corrects / len(train_loader.dataset)
        print("Train Loss: {}, Train Acc: {}".format(train_loss, train_acc))
        with open(result_path + "result.txt", "a", encoding="utf-8") as f:
            f.write("epoch: {}\n".format(epoch))
            f.write("Train Loss: {}, Train Acc: {}\n".format(train_loss, train_acc))
        test(model, test_loader, loss_func, result_path, epoch)
        torch.save(model.state_dict(), 'model/bilstm.pth')


def test(model, test_loader, loss_func, result_path, epoch):
    model.eval()
    with torch.no_grad():
        loss_val = 0.0
        corrects = 0.0
        label_true = []
        label_pred = []
        for datas, labels in test_loader:
            datas = datas.to(device)
            labels = labels.to(device)
            preds, preds_to_write = model(datas)
            if epoch > 3:
                with open(result_path + "output" + str(epoch) +".txt", 'a', encoding='utf-8') as f:
                    for l, p in zip(labels.tolist(), preds_to_write.tolist()):
                        f.write(str(l[1]) + " " + str(p[0]) + " " + str(p[1]) + "\n")

            loss = loss_func(preds, labels)
            loss_val += loss.item() * datas.size(0)
            labels = torch.argmax(labels, dim=1)
            preds = torch.argmax(preds, dim=1)
            corrects += torch.sum(preds == labels).item()

            label_true = label_true + labels.tolist()
            label_pred = label_pred + preds.tolist()

        test_loss = loss_val / len(test_loader.dataset)
        test_acc = corrects / len(test_loader.dataset)
        print("Test Loss: {}, Test Acc: {}".format(test_loss, test_acc))

        measure_result = classification_report(label_true, label_pred)
        print('measure_result = \n', measure_result)
        print("accuracy：%.4f" % accuracy_score(label_true, label_pred))
        print("precision：%.4f" % precision_score(label_true, label_pred))
        print("recall：%.4f" % recall_score(label_true, label_pred))
        print("f1-score：%.4f" % f1_score(label_true, label_pred))
        with open(result_path + "result.txt", 'a', encoding='utf-8') as f:
            f.write("epoch: %d\n" % epoch)
            f.write("accuracy：%.2f\n" % accuracy_score(label_true, label_pred))
            f.write("precision：%.2f\n" % precision_score(label_true, label_pred))
            f.write("recall：%.2f\n" % recall_score(label_true, label_pred))
            f.write("f1-score：%.2f\n" % f1_score(label_true, label_pred))

        with open(result_path + "result.txt", "a", encoding="utf-8") as f:
            f.write("Test Loss: {}, Test Acc: {}\n".format(test_loss, test_acc))
    return test_acc


if __name__ == "__main__":
    # vocab_size：word2Vec的长度
    # embedding_size：将每条数据的每一个字词展开的维度
    # hidden_dim：隐藏层特征数量，即h的维度
    # num_layers：lstm层数
    # num_directions：单向、双向
    # num_class：输出的维度
    # batch_size：每次获取的数据数量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_size = 16
    hidden_dim = 8
    num_layers = 1
    num_directions = 2
    num_class = 2
    batch_size = 16
    learning_rate = 0.001
    epochs = 10

    for i in range(8, 9):
        print("------------------------------ndata_" + str(i) + " -------------------------------------")
        train_path = "data/CrossValidation/ndata_" + str(i) + "/train.csv"
        test_path = "data/CrossValidation/ndata_" + str(i) + "/test.csv"

        vocab_size, sentence_maxlen = data_prepare(train_path, test_path)
        sentence_maxlen = int(sentence_maxlen * 0.9)
        print(vocab_size, sentence_maxlen)
        test_dataset = get_data(test_path, vocab_size, sentence_maxlen, embedding_size)
        train_dataset = get_data(train_path, vocab_size, sentence_maxlen, embedding_size)
        train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        model = BiLSTM(vocab_size=vocab_size, embedding_size=embedding_size, hidden_dim=hidden_dim,
                       num_layers=num_layers, num_directions=num_directions, num_class=num_class)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_func = nn.BCELoss()
        train(model, epochs, train_dataset, test_dataset, optimizer, loss_func, "data/CrossValidation/ndata_" + str(i) + "/")
        torch.save(model, "model/N折_" + str(i) + "_model.pth")
    #
    # for i in range(5, 50, 5):
    #     print("------------------------------robust_" + str(i) + " -------------------------------------")
    #     train_path = "data/Robust/robust_" + str(i) + "%/train.csv"
    #     test_path = "data/Robust/robust_" + str(i) + "%/test.csv"
    #
    #     vocab_size, sentence_maxlen = data_prepare(train_path, test_path)
    #     sentence_maxlen = int(sentence_maxlen * 0.9)
    #     print(vocab_size, sentence_maxlen)
    #     test_dataset = get_data(test_path, vocab_size, sentence_maxlen, embedding_size)
    #     train_dataset = get_data(train_path, vocab_size, sentence_maxlen, embedding_size)
    #     train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #     test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    #
    #     model = BiLSTM(vocab_size=vocab_size, embedding_size=embedding_size, hidden_dim=hidden_dim,
    #                    num_layers=num_layers, num_directions=num_directions, num_class=num_class)
    #     model = model.to(device)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #     loss_func = nn.BCELoss()
    #     train(model, epochs, train_dataset, test_dataset, optimizer, loss_func,
    #           "data/Robust/robust_" + str(i) + "%/")
    #     torch.save(model, "model/Robust" + str(i) + "%_model.pth")

    # for i in range(1, 6):
    #     print("------------------------------scale_" + str(i) + " -------------------------------------")
    #     train_path = "data/Scale/data_0-1to1-" + str(i) + "/train.csv"
    #     test_path = "data/Scale/data_0-1to1-" + str(i) + "/test.csv"
    #
    #     vocab_size, sentence_maxlen = data_prepare(train_path, test_path)
    #     sentence_maxlen = int(sentence_maxlen * 0.9)
    #     print(vocab_size, sentence_maxlen)
    #     test_dataset = get_data(test_path, vocab_size, sentence_maxlen, embedding_size)
    #     train_dataset = get_data(train_path, vocab_size, sentence_maxlen, embedding_size)
    #     train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #     test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    #
    #     model = BiLSTM(vocab_size=vocab_size, embedding_size=embedding_size, hidden_dim=hidden_dim,
    #                    num_layers=num_layers, num_directions=num_directions, num_class=num_class)
    #     model = model.to(device)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #     loss_func = nn.BCELoss()
    #     train(model, epochs, train_dataset, test_dataset, optimizer, loss_func,
    #           "data/Scale/data_0-1to1-" + str(i) + "/")
    #     torch.save(model, "model/Scale_0-1to1-" + str(i) + "-model.pth")
    # for i in range(2, 6):
    #     print("------------------------------scale_" + str(i) + " -------------------------------------")
    #     train_path = "data/Scale/data_0-1to" + str(i) + "-1/train.csv"
    #     test_path = "data/Scale/data_0-1to" + str(i) + "-1/test.csv"
    #
    #     vocab_size, sentence_maxlen = data_prepare(train_path, test_path)
    #     sentence_maxlen = int(sentence_maxlen * 0.9)
    #     print(vocab_size, sentence_maxlen)
    #     test_dataset = get_data(test_path, vocab_size, sentence_maxlen, embedding_size)
    #     train_dataset = get_data(train_path, vocab_size, sentence_maxlen, embedding_size)
    #     train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #     test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    #
    #     model = BiLSTM(vocab_size=vocab_size, embedding_size=embedding_size, hidden_dim=hidden_dim,
    #                    num_layers=num_layers, num_directions=num_directions, num_class=num_class)
    #     model = model.to(device)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #     loss_func = nn.BCELoss()
    #     train(model, epochs, train_dataset, test_dataset, optimizer, loss_func,
    #           "data/Scale/data_0-1to" + str(i) + "-1/")
    #     torch.save(model, "model/Scale_0-1to" + str(i) + "-1-model.pth")
