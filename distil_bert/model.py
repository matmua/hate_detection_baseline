import torch
from torch import nn
from transformers import DistilBertModel
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#加载预训练模型
pretrained = DistilBertModel.from_pretrained('D:\资料\python\项目\Bert_Lstm-main\Bert_Lstm-main\\distilbert-mult')
pretrained.to(device)
# 不训练,不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)

class DistilModel(nn.Module):
    def __init__(self):
        super(DistilModel, self).__init__()
        self.linear_sentence = nn.Linear(768, 64)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        self.linear_end = nn.Linear(64, 2)

    def forward(self, input_ids):
        bert_feature1 = self.linear_sentence(input_ids)
        #print(bert_feature1.shape)
        bert_feature = self.relu(bert_feature1)
        bert_feature = self.dropout(bert_feature)
        out = self.linear_end(bert_feature)
        return out
