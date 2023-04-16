
import torch
import torch.nn as nn

embed = nn.Embedding(100, 4)
lstm = nn.LSTM(input_size=4,
               hidden_size=16,
               num_layers=1,
               bidirectional=True,
               batch_first=True)
line = nn.Linear(16 * 2, 2)
acfun = nn.Softmax(dim=1)
x = torch.zeros(45, 20).long()
y = embed(x)
z, (h, c) = lstm(y)
out = torch.cat([h[0, :, :], h[1, :, :]], dim=-1)
final = line(out)
print(x.shape)
print(y.shape)
print(z.shape)
print(h.shape)
print(c.shape)
print(out.shape)
print(final.shape)
print(final)
for i in final.tolist():
    print(i)
final = acfun(final)
print(final.shape)
