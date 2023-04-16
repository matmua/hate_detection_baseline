
import torch
import torch.nn as nn

embed = nn.Embedding(100, 4)
lstm = nn.LSTM(input_size=4,
               hidden_size=16,
               num_layers=1,
               bidirectional=False,
               batch_first=True)
line = nn.Linear(16, 2)
acfun = nn.Softmax(dim=1)
x = torch.zeros(45, 20).long()
y = embed(x)
z, (h, c) = lstm(y)
h = h[0]
final = line(h)
print(x.shape)
print(y.shape)
print(z.shape)
print(h.shape)
print(c.shape)
print(final.shape)
final = acfun(final)
print(final.shape)
