import torch
import torch.nn as nn

embed = nn.Embedding(100, 4)
lstm = nn.GRU(input_size=4,
              hidden_size=16,
              num_layers=1,
              bidirectional=True,
              batch_first=True)
line = nn.Linear(16, 2)
acfun = nn.Softmax(dim=1)
x = torch.zeros(45, 20).long()
print(x.shape)
y = embed(x)
print(y.shape)
z, h = lstm(y)
print(z.shape)
print(h.shape)
final = line(h)
print(final.shape)
print(final)
final = acfun(final)
print(final.shape)
