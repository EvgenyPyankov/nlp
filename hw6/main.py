import numpy as np
from sklearn.cluster import KMeans, AffinityPropagation, Birch
from sklearn.metrics import homogeneity_score, f1_score
from sklearn.model_selection import train_test_split

from datasets_helper import DatasetsHelper


def split_dataset(dataset, train_size):
    attributes = dataset.values[:, 0:-1]
    classes = dataset.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(attributes, classes, train_size=train_size)
    return X_train, X_test, y_train, y_test

def compare(y, y_pred):
    dict = {}
    for i in range(0, len(y)):
        dict.setdefault(y[i], []).append(y_pred[i])
    print(dict)


def kmean(X_train, X_test, y_train, y_test):
    print("\nKMean:")
    n_clusters = len(np.unique(y_train))
    print("Number of clusters: %d" % n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_train)
    y_pred = kmeans.predict(X_test)
    print("Homogenity score: %0.3f" % homogeneity_score(y_test, y_pred))
    compare(y_test, y_pred)


def affinity_propagation(X_train, X_test, y_train, y_test):
    print("\nAffinityPropagation:")
    af = AffinityPropagation().fit(X_test)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    n_clusters_ = len(cluster_centers_indices)

    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity score: %0.3f" % homogeneity_score(y_test, labels))
    y_pred = af.predict(X_test)
    compare(y_test, y_pred)

def birch(X_train, X_test, y_train, y_test):
    print("\nBirch:")
    n_clusters = len(np.unique(y_train))
    print("Number of clusters: %d" % n_clusters)
    brc = Birch(n_clusters=n_clusters)
    brc.fit(X_train)
    y_pred = brc.predict(X_test)
    print("Homogenity score: %0.3f" % homogeneity_score(y_test, y_pred))
    compare(y_test, y_pred)


datasetsHelper = DatasetsHelper()
# datasetsHelper.prepare_dataset()

dataset = datasetsHelper.load_dataset('dataset_events.csv')

X_train, X_test, y_train, y_test = split_dataset(dataset, train_size=0.7)

kmean(X_train, X_test, y_train, y_test)
affinity_propagation(X_train, X_test, y_train, y_test)
birch(X_train, X_test, y_train, y_test)
