import re
from nltk.tokenize import word_tokenize

# Reading file
filename = 'data/extracted_text.txt'
file = open(filename, 'rt', encoding='UTF8')
text = file.readlines()
file.close()

# Delete punctuation
filtered_sentences = []
counter = 0
for line in text:
    sent = ""
    tokens = word_tokenize(line)
    for token in tokens:
        if token.isalpha() or re.search('\w+-\w+', token) != None:
            sent = sent + " " + token
    filtered_sentences.append(sent.strip())
    counter += 1
    print("Обработано предложение №" + str(counter))

# Writing file
output_file_name = 'data/cleaned_text.txt'
output_file = open(output_file_name, 'wt', encoding='UTF8')
output_file.writelines(line + '\n' for line in filtered_sentences)
output_file.close()