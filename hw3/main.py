import string

import gensim
from gensim.models.word2vec import LineSentence
from nltk.corpus import stopwords


def train_model(sentances):
    return gensim.models.Word2Vec(sentances, min_count=5, size=300, workers=4, window=10, sg=1, negative=5)


sentences = LineSentence('../wuthering_heights.txt')
model = train_model(sentences)

model.wv.save_word2vec_format("heights_model.model", binary=False)

print(model.most_similar(positive=['house']))

# Analogies
print(model.most_similar(positive=['man', 'Heathcliff'], negative=['woman']))

# Doesn't match
print(model.doesnt_match(['Hindley', 'Catherine', 'Edgar', 'Hareton']))
