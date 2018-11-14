import nltk
from nltk import *
from nltk.corpus import stopwords

# 1.Download the text you selected
filename = "../wuthering_heights.txt"
with open(filename, "r", encoding="utf8") as file:
    raw = file.read()

# 2.Apply word and sentence tokenization
words = nltk.word_tokenize(raw)
sents = nltk.sent_tokenize(raw)

# 3.Convert to a nltk Text
text = nltk.Text(words)

# 4.Use NLTK FreqDist to print and plot the most common words in your book
fdist_heights = FreqDist(text)
most_common_heights = fdist_heights.most_common(50)
print("Most common words of Wuthering Heights:\n" + str(most_common_heights))
fdist_heights.plot(20, title="Most common words of Wuthering Heights", cumulative=False)

# 5.Compare the frequency to “Moby Dick” (text1) book in NLTK
from nltk.book import text1
fdist_moby = FreqDist(text1)
most_common_moby = fdist_moby.most_common(50)
print("Most common words of Moby Dick:\n" + str(most_common_moby))
fdist_moby.plot(20, title="Most common words of Moby Dick", cumulative=False)

# 6. Deleting stop words
stop_words = stopwords.words("english")
filtered_words = []
for word in words:
    if word.isalpha() and word not in stop_words:
        filtered_words.append(word)

fdist_heights_filtered = FreqDist(filtered_words)
most_common_heights_filtered = fdist_heights_filtered.most_common(50)
print("Most common words of Wuthering Heights (filtered):\n" + str(most_common_heights_filtered))
fdist_heights_filtered.plot(20, title="Most common words of Wuthering Heights (filtered)", cumulative=False)

# 7.Split your assigned gutenberg book into paragraphs
n = 30
paragraphs = []
for i in range(0, len(sents), n):
    paragraph = sents[i:i + n]
    paragraphs.append(paragraph)

# 8.Now create an positional index
terms_list = []
for i, paragraph in enumerate(paragraphs):
    tokens = word_tokenize(' '.join(paragraph))
    for j, token in enumerate(tokens):
        terms_list.append((token, i, j))
terms_list = sorted(terms_list)

dictionary = {}
for (term, document_num, position) in terms_list:
    if term not in dictionary:
        dictionary[term] = {'postings': dict()}
    dictionary[term]['postings'].setdefault(document_num, []).append(position)

print(dictionary)