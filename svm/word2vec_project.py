#import gensim.models
#from gensim.models import Word2Vec

#sentences=Word2Vec.Text8Corpus("ntest.txt")
#model=gensim.models.Word2Vec(sentences,sg=0,size=100,window=5,min_count=2,negative=3,sample=0.001,hs=1,workers=4)
#model = Word2Vec(sentences, sg=0, size=100,  window=5,  min_count=2,  negative=3, sample=0.001, hs=1, workers=4)
#model.wv.save_word2vec_format("result.txt",binary=False)

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.test.utils import common_texts, get_tmpfile

# inp为输入语料
inp = 'ntest.txt'
sentences = LineSentence(inp)
#path = get_tmpfile("word2vec.model")  # 创建临时文件
model = Word2Vec(sentences, vector_size=100, sg=0,  window=5,  min_count=2,  negative=3, sample=0.001, hs=1, workers=4)
model.save("word2vec.pkl")
#model.wv.save_word2vec_format("result.txt",binary=False)