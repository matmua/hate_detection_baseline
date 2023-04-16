import torch
import data_process
import model
from transformers import AdamW
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_curve = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 定义一些参数，模型选择了最基础的bert中文模型
batch_size = 16
epoches = 80
hidden_size = 768
n_class = 2
maxlen = 80

encode_layer = 12
filter_sizes = [2, 2, 2]
num_filters = 3

#数据加载
loader = data_process.dataprocess()


meanmodel = model.Bert_Blend_CNN()
#meanmodel.load_state_dict(torch.load('blend_param.pth'))
meanmodel.to(device)



#训练
optimizer = AdamW(meanmodel.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()

meanmodel.train()
for epoch in range(3):
    print("epoch数：" + str(epoch+1))
    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader):
        #print(input_ids)
        #print(labels)
        if (torch.cuda.is_available()):
            input_ids, attention_mask, token_type_ids,labels = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device),labels.to(device)
        out = meanmodel([input_ids, attention_mask, token_type_ids])
        #print(out)
        #print(feature.shape)
        #print("out:" + str(out))
        #print("label:" + str(labels))
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 5 == 0:
            out = out.argmax(dim=1)
            accuracy = (out == labels).sum().item() / len(labels)

            print(i, loss.item(), accuracy)

        if i == 125:#2000条
            break

torch.save(meanmodel.state_dict(), './blend_param.pth')
