import csv
import glob
import json

import pandas as pnd
from sklearn.feature_extraction.text import TfidfVectorizer

class DatasetsHelper:

    def load_dataset(self, path):
        return pnd.read_csv(path, header=None)

    def preprocess(self, text):
        return text

    def get_all_documents(self, path):
        all_documents = {}
        for file_name in glob.glob(path):
            with open(file_name, encoding='utf8') as file:
                text = file.read()
                all_documents[file_name] = self.preprocess(text)
        return all_documents

    def get_vectors(self, all_documents):
        vectorizer = TfidfVectorizer()
        td_matrix = vectorizer.fit_transform(all_documents.values()).toarray()

        file_name_list = all_documents.keys()
        vectors = {}

        for index, file_name in enumerate(file_name_list):
            vectors[file_name] = td_matrix[index]
        return vectors

    def prepare_csv_metadata(self, dataset_name, file_name):
        metadata_file_name = file_name.replace('.txt', '_metadata.json')
        with open(metadata_file_name, encoding='utf8') as file:
            metadata = json.load(file)
        return metadata, file_name

    def write_event_steps_dataset(self, dataset_name, vectors):
        with open(dataset_name, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for file_name in vectors:
                metadata, file_name = self.prepare_csv_metadata(dataset_name, file_name)
                row = []
                row.extend(vectors[file_name])
                row.append(metadata["general_topic"] + ':' + metadata["topic"])
                spamwriter.writerow(row)

    def write_events_dataset(self, dataset_name, vectors):
        with open(dataset_name, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for file_name in vectors:
                metadata, file_name = self.prepare_csv_metadata(dataset_name, file_name)
                row = []
                row.extend(vectors[file_name])
                row.append(metadata["general_topic"])
                spamwriter.writerow(row)

    def prepare_dataset(self):
        documents = self.get_all_documents("data/*.txt")
        vectors = self.get_vectors(documents)
        self.write_event_steps_dataset("dataset_event_steps.csv", vectors)
        self.write_events_dataset("dataset_events.csv", vectors)
