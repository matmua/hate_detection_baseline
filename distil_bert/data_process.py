import torch
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertModel
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载字典和分词工具
token = DistilBertTokenizer.from_pretrained('bert-base-chinese')
print("token:")
print(token)

#加载预训练模型
pretrained = DistilBertModel.from_pretrained('D:\资料\python\项目\Bert_Lstm-main\Bert_Lstm-main\\distilbert-mult')
pretrained.to(device)
# 不训练,不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)

#定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split, data_f, dir):
        #self.dataset = load_dataset(path='lansinuote/ChnSentiCorp', split=split)
        self.dataset = load_dataset("csv", data_dir=dir,
                                    data_files=data_f, split=split)
        #"test_with_label.csv"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['Sentence']
        label = self.dataset[i]['Label']
        return text, label

def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    #print(sents[0:16],labels[0:16])
    #编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=80,
                                   return_tensors='pt',
                                   return_length=True)
    #input_ids:编码之后的数字
    #attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']
    input_ids = input_ids.to(device)
    with torch.no_grad():
        last_hidden_states = pretrained(input_ids)
    out = last_hidden_states.last_hidden_state[:, 0]
    #attention_mask = data['attention_mask']
    #token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)
    #print(data['length'], data['length'].max())
    return out, labels

def dataprocess(data_f, dir):
    dataset = Dataset('train', data_f, dir)
    print(len(dataset), dataset[0])
    # 数据加载器
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=32,
                                         collate_fn=collate_fn,
                                         shuffle=True,
                                         drop_last=True)
    print(len(loader))
    return loader