import torch
import torchvision

import jieba

txt = open("test.txt",encoding='UTF-8').read()
words = jieba.lcut(txt)
result = open('ntest.txt','w',encoding='UTF-8')
result.write('  '.join(words))
counts = {}
for word in words:
    if len(word) == 1:
        continue
    else:
        counts[word] = counts.get(word,0) + 1
items = list(counts.items())
items.sort(key=lambda x:x[1], reverse=True)
for i in range(30):
  word, count = items[i]
  print (u"{0:<10}{1:>5}".format(word, count))
  result.write((str(items[i])))
result.close()
