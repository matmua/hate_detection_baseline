import torch
import data_process
import model
from transformers import AdamW
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#数据加载
loader = data_process.dataprocess()
for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
    break
input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device)

#bert测试
def berttest():
    # 模型试算
    pretrained = model.pretrained
    out = pretrained(input_ids=input_ids,
                     attention_mask=attention_mask,
                     token_type_ids=token_type_ids)
    print("out.last_hidden_state.shape:")
    print(out.last_hidden_state.shape)

berttest()

meanmodel = model.MeanModel()
#meanmodel.load_state_dict(torch.load('hate_param.pth'))
meanmodel.to(device)

'''print(meanmodel(input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids).shape)'''


#训练
optimizer = AdamW(meanmodel.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()

meanmodel.train()
for epoch in range(2):
    print("epoch数：" + str(epoch+1))
    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader):
        #print(input_ids)
        #print(labels)
        if (torch.cuda.is_available()):
            input_ids, attention_mask, token_type_ids,labels = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device),labels.to(device)
        out, feature = meanmodel(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        #print(out)
        #print(feature.shape)
        print("out:" + str(out))
        print("label:" + str(labels))
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

torch.save(meanmodel.state_dict(), './hate_param.pth')
