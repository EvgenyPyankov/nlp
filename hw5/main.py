import difflib
from difflib import SequenceMatcher
import nltk
import spacy
from nltk import *


def replace_prp(ne_tree):
    tokens = []
    previous = None
    for node in ne_tree:
        if type(node) is nltk.Tree:
            tokens.append(' ')
            previous = node[0][0]
            tokens.append(previous)
        else:
            if not (node[1] == '.' or node[1] == ','):
                tokens.append(' ')
            if (node[1] == "PRP"):
                tokens.append(previous)
            else:
                tokens.append(node[0])

    str = ''.join(tokens)
    return str.strip()


def get_diff(first, second):
    result = ""
    for line in difflib.unified_diff(first, second, fromfile='first', tofile='second', lineterm=''):
        result = result + line + "\n"
    result = "The two texts are identical" if result == "" else result
    return result


# Have some short input document
source = "Andrew is a software engineer. He works at Google. Ada is a manager. She works at Google as well."
expected = "Andrew is a software engineer. Andrew works at Google. Ada is a manager. Ada works at Google as well."

# Apply POS-tagging to the document
text = word_tokenize(source)
tagged = nltk.pos_tag(text)
print("NLTK tagged text:\n" + str(tagged))

# Apply NE-Recognition to the document
ne_tree = nltk.ne_chunk(tagged)
print("NLTK NE-Recognition:\n" + str(ne_tree))

# For any pronoun replace it with previous Named Entity
replaced = replace_prp(ne_tree)

# Print to result text
print("My system replaced:\n" + str(replaced))

# Evaluate your system
diff = get_diff(replaced, expected)
matcher = SequenceMatcher(None, replaced, expected)
print("Diff between expected and my system result:\n" + str(diff)
      + "\nratio = " + str(matcher.ratio()))

# Run spaCy coref on the same text
nlp = spacy.load('en_coref_md')
doc = nlp(source)
result_spacy = doc._.coref_resolved
print("spaCy replace:\n" + str(result_spacy))
diff = get_diff(str(result_spacy), expected)
matcher = SequenceMatcher(None, str(result_spacy), expected)
print("Diff between expected and spaCy result:\n" + str(diff)
      + "\nration = " + str(matcher.ratio()))
