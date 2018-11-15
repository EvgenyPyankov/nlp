import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
from gensim.models.word2vec import LineSentence
lmtzr = WordNetLemmatizer()
tokenize_sentences = nltk.sent_tokenize(open('../wuthering_heights.txt', encoding= 'utf - 8').read())
result = ""
for sentence in tokenize_sentences:
    words_raw = nltk.word_tokenize(sentence)
    words = [word for word in words_raw if word.isalpha()]
    if len(words) > 0:
        clear_sentence = ''
        for word in words:
            clear_sentence += lmtzr.lemmatize(word) + ' '
        clear_sentence += '. '
        result += clear_sentence
clear_file = open("all_clear_text.txt",'wt', encoding='utf-8')
clear_file.writelines(result)

def train_model(text):
    return gensim.models.Word2Vec(text, min_count=5, size=300, workers=4, window=10, sg=1, negative=5)
clear_model = train_model(LineSentence("all_clear_text.txt"))
clear_model.wv.save_word2vec_format("heights_clear.model", binary=False)
