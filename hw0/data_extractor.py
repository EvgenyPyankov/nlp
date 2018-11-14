import re

file_name = "data/ru_ar_cut.txt"

sentences = []
with open(file_name, 'r', encoding='UTF8') as file:
    sent = ""
    for line in file:
        word = line.split(None, 1)[0]
        if word == "</s>":
            sentences.append(sent.strip())
            sent = ""
        else:
            if re.match("<.*", word) == None:
                sent = sent + " " + word


output_file_name = 'data/extracted_text.txt'
output_file = open(output_file_name, 'wt', encoding='UTF8')
output_file.writelines(line + '\n' for line in sentences)
output_file.close()

