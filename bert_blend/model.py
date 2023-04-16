import torch
from torch import nn
from transformers import BertModel, AutoModel, AutoTokenizer
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#加载预训练模型
pretrained = AutoModel.from_pretrained('D:\资料\python\项目\Bert_Lstm-main\Bert_Lstm-main\\bert-base-chinese', output_hidden_states=True, return_dict=True)
pretrained.to(device)
# 不训练,不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)

train_curve = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 定义一些参数，模型选择了最基础的bert中文模型
batch_size = 16
epoches = 80
hidden_size = 768
n_class = 2
maxlen = 8

encode_layer = 12
filter_sizes = [2, 2, 2]
num_filters = 3

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.num_filter_total = num_filters * len(filter_sizes)
        self.Weight = nn.Linear(self.num_filter_total, n_class, bias=False)
        self.bias = nn.Parameter(torch.ones([n_class]))
        self.filter_list = nn.ModuleList([
            nn.Conv2d(1, num_filters, kernel_size=(size, hidden_size)) for size in filter_sizes
        ])

    def forward(self, x):
        # x: [bs, seq, hidden]
        x = x.unsqueeze(1)  # [bs, channel=1, seq, hidden]

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            h = F.relu(conv(x))  # [bs, channel=1, seq-kernel_size+1, 1]
            mp = nn.MaxPool2d(
                kernel_size=(encode_layer - filter_sizes[i] + 1, 1)
            )
            # mp: [bs, channel=3, w, h]
            pooled = mp(h).permute(0, 3, 2, 1)  # [bs, h=1, w=1, channel=3]
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, len(filter_sizes))  # [bs, h=1, w=1, channel=3 * 3]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])

        output = self.Weight(h_pool_flat) + self.bias  # [bs, n_class]

        return output


class Bert_Blend_CNN(nn.Module):
    def __init__(self):
        super(Bert_Blend_CNN, self).__init__()
        self.bert = pretrained
        self.linear = nn.Linear(hidden_size, n_class)
        self.textcnn = TextCNN()

    def forward(self, X):
        input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        '''print("input_ids:" + str(input_ids.shape))
        print("attention_mask:" + str(attention_mask.shape))
        print("token_type_ids:" + str(token_type_ids.shape))'''
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # 返回一个output字典
        # 取每一层encode出来的向量
        # outputs.pooler_output: [bs, hidden_size]
        hidden_states = outputs.hidden_states  # 13*[bs, seq_len, hidden] 第一层是embedding层不需要
        #print(hidden_states[1].shape)
        cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1)  # [bs, 1, hidden]
        # 将每一层的第一个token(cls向量)提取出来，拼在一起当作textcnn的输入
        for i in range(2, 13):
            cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
        # cls_embeddings: [bs, encode_layer=12, hidden]
        logits = self.textcnn(cls_embeddings)
        return logits