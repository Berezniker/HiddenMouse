from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.utils import shuffle
from sklearn import svm
import numpy as np
import pandas as pd
import os

N_FEATURES = 16
RANDOM_SEED = 13


def data_preprocessing(dir_name, labels):
    X = np.empty((labels.index.size, N_FEATURES))
    y = np.empty(labels.index.size)
    # TODO
    return X, y


def fit(X):
    model = svm.LinearSVC(C=1.6)
    #_X_, _y_ = shuffle(X, y, random_state=RANDOM_SEED)
    model.fit(X)
    return model


def classify(model, y):
    return model.predict(y)


if __name__ == '__main__':
    features_path = f'../features'
    labels_path = f'../dataset/labels.csv'

    labels = pd.read_csv(labels_path)
    train_features, train_labels = data_preprocessing(os.path.join(features_path, 'train_features'), labels)
    # model = fit(train_features)

    test_features, test_labels = data_preprocessing(os.path.join(features_path, 'test_features'), labels)
    # prediction = classify(model, test_features)
    # metric = np.mean(test_features == prediction)
    # print("Ok,", metric)
