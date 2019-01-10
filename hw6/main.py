import pandas as pnd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from data import Data

csv = pnd.read_csv('data/iris.csv')
data = Data(csv, 0.7)

sklearn_nb_clf = GaussianNB()
sklearn_nb_clf.fit(data.train_x, data.train_y)
sklearn_nb_clf_accuracy = accuracy_score(data.test_y, sklearn_nb_clf.predict(data.test_x))
print(sklearn_nb_clf_accuracy)
