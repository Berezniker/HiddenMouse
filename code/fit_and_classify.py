from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import svm
import numpy as np
import pandas as pd
import glob
import time
import os

N_FEATURES = 16
RANDOM_SEED = 13
SVM_KERNEL = 'rbf'  # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
SVM_DEGREE = 3      # Degree of the 'ploy'. Ignored by all other kernels.
SVM_GAMMA = 'auto'  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'


def get_session_path(data_path, labels=None):
    find_session = {}
    for user in glob.glob(os.path.join(data_path, 'user*')):
        user_name = os.path.basename(user)
        for session in glob.glob(os.path.join(user, 'session*')):
            if labels is None:
                find_session.setdefault(user_name, []).append(session)
            elif os.path.basename(session) in labels.filename.values:
                find_session.setdefault(user_name, []).append(session)

    return find_session


def split_into_legal(test_session, labels):
    legal_session, illegal_session = {}, {}
    for user, user_sessions in test_session.items():
        for session in user_sessions:
            if labels[labels.filename == os.path.basename(session)].is_illegal.values:
                illegal_session.setdefault(user, []).append(session)
            else:
                legal_session.setdefault(user, []).append(session)

    return legal_session, illegal_session


def get_users_svm(session_dict, only_one_user=True):
    users_svm = {}
    print('> SVM.fit')
    svm_start_time = time.time()
    for user, user_session in session_dict.items():
        print(f'>> {user}')
        user_start_time = time.time()
        users_svm[user] = fit_svm(user_session)
        print(f'>> {user} time: {time.time() - user_start_time:.3f} sec')
        if only_one_user:
            print('>> [!] only_one_user')
            break
    print(f'> SVM.fit time: {time.time() - svm_start_time:.3f} sec')
    return users_svm


def fit_svm(features_path):
    features = None
    for feature_path in features_path:
        f = pd.read_csv(feature_path, sep=',', header=0).values
        features = np.vstack((features, f)) if features is not None else f
    print(f'>>> features.shape={features.shape}')

    # https://scikit-learn.org/0.15/modules/generated/sklearn.svm.OneClassSVM.html
    model = svm.OneClassSVM(kernel=SVM_KERNEL,
                            degree=SVM_DEGREE,
                            gamma=SVM_GAMMA)
    model.fit(features)
    return model


def classify(users_svm, train_session, test_legal_session, test_illegal_session):
    pass


if __name__ == '__main__':
    train_features_path = r'../features/train_features'
    test_features_path = r'../features/test_features'
    labels_path = r'../dataset/labels.csv'

    print('RUN')
    start_time = time.time()

    labels = pd.read_csv(labels_path)
    train_sessions = get_session_path(train_features_path)
    users_svm = get_users_svm(train_sessions, only_one_user=True)

    test_sessions = get_session_path(test_features_path, labels)
    test_legal_sessions, test_illegal_sessions = split_into_legal(test_sessions, labels)
    # classify(users_svm, train_sessions, test_legal_sessions, test_illegal_sessions)

    print(f'run time: {time.time() - start_time:.3f}')
