from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import glob
import time
import os

# TODO [READ!] https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
# TODO [READ!] https://scikit-learn.org/stable/modules/manifold.html#manifold

"""
https://dyakonov.org/2017/04/19/%D0%BF%D0%BE%D0%B8%D1%81%D0%BA-%D0%B0%D0%BD%D0%BE%D0%BC%D0%B0%D0%BB%D0%B8%D0%B9-anomaly-detection/

https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html
"""

N_FEATURES = 81  # 119
ALL_TEST_FEATURES = None
USERS_FEATURES_SLICE = dict()
# KEEP_LOG in Unix: python3 fit_and_classify.py | tee {LOG_FILE_NAME}

COLOR = {
    'yellow': '\033[93m',
    'green': '\033[92m',
    'cyan': '\033[96m',
    'red': '\033[91m',
    'None': '\033[0m'
}


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


def load_features(features_path):
    features = None
    for feature_path in features_path:
        f = pd.read_csv(feature_path, sep=',', header=0).values
        features = np.vstack((features, f)) if features is not None else f
    # return features
    return features[:, :32]


def load_all_test_features(test_sessions):
    global ALL_TEST_FEATURES, USERS_FEATURES_SLICE
    for user in test_sessions:
        user_features = load_features(test_sessions[user])
        if ALL_TEST_FEATURES is None:
            USERS_FEATURES_SLICE[user] = slice(0, user_features.shape[0])
        else:
            offset = ALL_TEST_FEATURES.shape[0]
            USERS_FEATURES_SLICE[user] = slice(offset, offset + user_features.shape[0])
        ALL_TEST_FEATURES = np.vstack((ALL_TEST_FEATURES, user_features)) \
            if ALL_TEST_FEATURES is not None else user_features
    ALL_TEST_FEATURES = ALL_TEST_FEATURES[:, :32]
    return ALL_TEST_FEATURES


def get_legal_test_features_for_user(user):
    global ALL_TEST_FEATURES, USERS_FEATURES_SLICE
    return ALL_TEST_FEATURES[USERS_FEATURES_SLICE[user]]


def get_all_illegal_test_features_for_user(user):
    global ALL_TEST_FEATURES, USERS_FEATURES_SLICE
    return np.delete(ALL_TEST_FEATURES, USERS_FEATURES_SLICE[user], axis=0)


def split_into_legal(test_session, labels):
    legal_session, illegal_session = dict(), dict()
    for user, user_sessions in test_session.items():
        for session in user_sessions:
            if labels[labels.filename == os.path.basename(session)].is_illegal.values:
                illegal_session.setdefault(user, []).append(session)
            else:
                legal_session.setdefault(user, []).append(session)

    return legal_session, illegal_session


# def get_pairwise_score(model, user_name: str, test_sessions, print_score=False):
#     auc, accuracy, frr, far = list(), list(), list(), list()
#     legal_features = get_legal_test_features_for_user(user_name)
#     # legal_features = legal_features[int(legal_features.shape[0] * 0.7):]
#     for other_user in test_sessions:
#         if other_user == user_name:
#             continue
#         illegal_features = get_legal_test_features_for_user(other_user)
#         pairwise_features = np.vstack((legal_features, illegal_features))
#         y_true = np.ones(pairwise_features.shape[0])
#         y_true[legal_features.shape[0]:] = -1
#         y_score = model.decision_function(pairwise_features)
#         # ^-- Signed distance is positive for an inlier and negative for an outlier.
#         auc.append(roc_auc_score(y_true, y_score))
#         y_score = model.predict(pairwise_features)
#         accuracy.append(np.mean(y_score == y_true))
#         frr.append(np.sum(y_score[:legal_features.shape[0]] == -1) / pairwise_features.shape[0])
#         far.append(np.sum(y_score[-illegal_features.shape[0]:] == 1) / pairwise_features.shape[0])
#         if print_score:
#             print(f">> {user_name} vs {other_user}:\n"
#                   f"    AUC     = {auc[-1]:.2f}\n"
#                   f"    ACC     = {accuracy[-1]:.2f}\n"
#                   f"    FRR(I)  = {frr[-1]:.2f}%\n"
#                   f"    FAR(II) = {far[-1]:.2f}%\n")
#     return np.mean(auc), np.mean(accuracy), np.mean(frr), np.mean(far)

SCALER = None


def get_pairwise_score(model, user_name, test_sessions, print_score=False):
    global SCALER
    auc, accuracy, frr, far = list(), list(), list(), list()
    legal_features = get_legal_test_features_for_user(user_name)
    # legal_features = legal_features[-int(legal_features.shape[0] * 0.3):]
    all_illegal_features = get_all_illegal_test_features_for_user(user_name)
    all_illegal_features = shuffle(all_illegal_features)

    n_fold = all_illegal_features.shape[0] // legal_features.shape[0]
    for illegal_features in np.array_split(all_illegal_features, n_fold):
        pairwise_features = np.vstack((legal_features, illegal_features))
        pairwise_features = SCALER.transform(pairwise_features)
        y_true = np.ones(pairwise_features.shape[0])
        y_true[legal_features.shape[0]:] = -1
        y_score = model.decision_function(pairwise_features)
        # ^-- Signed distance is positive for an inlier and negative for an outlier.
        auc.append(roc_auc_score(y_true, y_score))
        y_score = model.predict(pairwise_features)
        accuracy.append(np.mean(y_score == y_true))
        frr.append(np.sum(y_score[:legal_features.shape[0]] == -1) / pairwise_features.shape[0])
        far.append(np.sum(y_score[-illegal_features.shape[0]:] == 1) / pairwise_features.shape[0])
        if print_score:
            print(f"    AUC     = {auc[-1]:.2f}\n"
                  f"    ACC     = {accuracy[-1]:.2f}\n"
                  f"    FRR(I)  = {frr[-1]:.2f}%\n"
                  f"    FAR(II) = {far[-1]:.2f}%\n")
        return np.mean(auc), np.mean(accuracy), np.mean(frr), np.mean(far)


def classic_model(user_name: str, user_features_path, test_sessions, mode: str):
    features = load_features(user_features_path)
    # features = features[:int(features.shape[0] * 0.7)]
    if mode == 'if':
        param_grid = ParameterGrid(param_grid={
            'n_estimators': [1],
            'max_samples': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            'contamination': ['auto', 0.5],
            'max_features': [1.0],
            'bootstrap': [True],
            # 'warm_start': [True, False]
        })
        # features_2sec:   auc = 0.70 +- 0.04, FRR = 14.4 ± 9.0 %, FAR = 18.3 ± 13.4 %
        # features_5sec:   auc = 0.70 +- 0.04, FRR = 12.2 ± 8.6 %, FAR = 18.6 ± 13.9 %
        # features_10sec:  auc = 0.71 +- 0.05, FRR = 14.7 ± 6.8 %, FAR = 15.9 ± 12.3 %
        # features_15sec:  auc = 0.71 +- 0.04, FRR = 15.6 ± 7.9 %, FAR = 16.0 ± 11.9 %
        # -----------------------------------------------------------------------------
        # features_2sec:   auc = x.xx +- x.xx, FRR = 21.5 ± 6.9 %, FAR = 25.1 ± 9.0 %
        # features2_5sec:  auc = 0.54 +- 0.01, FRR = 22.3 ± 6.7 %, FAR = 23.8 ± 8.8 %
        # features2_10sec: auc = 0.55 +- 0.01, FRR = 23.2 ± 6.2 %, FAR = 22.3 ± 8.1 %
        # features2_15sec: auc = 0.56 +- 0.01, FRR = 23.7 ± 6.7 %, FAR = 22.7 ± 8.8 %
    elif mode == 'ee':
        param_grid = ParameterGrid(param_grid={
            'store_precision': [True, False],
            'assume_centered': [True],
            'support_fraction': [1., 0.9],
            'contamination': [0.3, 0.2, 0.1],
        })
        # features_5sec:   auc = 0.69 +- 0.05, FRR = 21.8 ± 9.8 %, FAR = 13.9 ± 10.9 %
        # features_10sec:  auc = 0.71 +- 0.05, FRR = 22.8 ± 7.6 %, FAR = 16.3 ± 10.6 %
        # features_15sec:  auc = 0.72 +- 0.05, FRR = 24.1 ± 8.2 %, FAR = 15.0 ± 11.3 %
        # ----------------------------------------------------------------------------
        # features2_2sec:  auc =
        # features2_5sec:  auc = 0.53 +- 0.01, FRR = 15.1 ± 1.8 %, FAR = 32.1 ± 12.7 %
        # features2_10sec: auc = 0.55 +- 0.01, FRR = 15.2 ± 1.8 %, FAR = 31.8 ± 11.9 %
        # features2_15sec: auc = 0.55 +- 0.01, FRR = 15.9 ± 4.2 %, FAR = 31.1 ± 10.6 %
    elif mode == 'locout':
        param_grid = ParameterGrid(param_grid={
            'n_neighbors': [10, 20],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
            'metric': ['minkowski', 'cityblock', 'manhattan'],
            'contamination': ['auto', 0.5, 0.4, 0.3, 0.2, 0.1],
            'novelty': [True]
        })
        # features_5sec:  FRR = 25.4 ± 11.4 %, FAR = 11.8 ± 9.2 %
        # features_10sec:
        # features_15sec:
        # -----------------------------------
        # features2_2sec:
        # features2_5sec:
        # features2_10sec:
        # features2_15sec: FRR = 24.0 ± 6.1 %, FAR = 20.5 ± 5.8 %
    else:  # mode == 'svm'
        param_grid = ParameterGrid(param_grid={
            'kernel': ['rbf'],
            'gamma': ['scale'],  # 'scale' ~ gamma=1/(n_features * X.var())
            'nu': np.linspace(0.20, 0.05, 16),  # [0.20, 0.15, 0.10, 0.208],
            'shrinking': [True]
            # 'degree', 'coef0', 'tol': ignored
        })
        # features_5sec:   nu: [0.25, 0.15], gamma: 'scale', auc=0.66 +- 0.04
        # features_10sec:  nu: [0.25, 0.15], gamma: 'scale', auc=0.67 +- 0.04, FRR = 14.5 ± 4.8 %, FAR = 54.2 ± 35.1 %
        # features_15sec:  nu: [0.25..0.10], gamma: 'scale', auc=0.67 +- 0.04, FRR = 14.7 ± 4.9 %, FAR = 52.9 ± 35.8 %
        # -------------------------------------------------------------------
        # features2_5sec:  nu: 0.20, gamma: 1e-4, acc=1.0б  FRR = 11.4 ± 2.9 %б FAR = 35.3 ± 8.7 %
        # features2_10sec: nu: 0.20, gamma: 1e-4, acc=1.0,  FRR = 11.7 ± 2.7 %, FAR = 32.1 ± 7.4 %
        # features2_15sec: nu: 0.208, gamma: 1e-4, auc=1.0, FRR = 12.8 ± 3.2 %, FAR = 28.5 ± 6.3 %

    best_param = (None, 0, 0, 0, 1, 1, ...)  # (model, train_score, auc, accuracy, FRR, FAR **param_grid)
    inliers = LocalOutlierFactor().fit_predict(X=features)
    features = features[np.argwhere(inliers == 1).squeeze()]

    global SCALER
    SCALER = StandardScaler().fit(features)
    features = SCALER.transform(features)

    for param in param_grid:
        model = OneClassSVM(**param) if mode == 'svm' else \
            (IsolationForest(**param) if mode == 'if' else
             (EllipticEnvelope(**param) if mode == 'ee' else
              (LocalOutlierFactor(**param))))
        model.fit(X=features)
        train_score = np.mean(model.predict(features) == 1)
        auc, acc, frr, far = get_pairwise_score(model, user_name, test_sessions)
        if (0.5 * frr + far) / 2 < (0.5 * best_param[4] + best_param[5]) / 2:
            if mode == 'svm':
                best_param = (model, train_score, auc, acc, frr, far, param['kernel'], param['nu'], param['gamma'])
            elif mode == 'if':
                best_param = (model, train_score, auc, acc, frr, far, param['n_estimators'], param['max_samples'],
                              param['contamination'], param['max_features'], param['bootstrap'])
            elif mode == 'ee':
                best_param = (model, train_score, auc, acc, frr, far, param['store_precision'],
                              param['assume_centered'], param['support_fraction'], param['contamination'])
            elif mode == 'locout':
                best_param = (model, train_score, auc, acc, frr, far, param['n_neighbors'], param['algorithm'],
                              param['metric'], param['contamination'], param['novelty'])
            print(COLOR['yellow'], end='')

        if True:
            if mode == 'if':
                print(f">>> train.score: {train_score:.4f}, "
                      f"auc = {auc:.3f}, "
                      f"acc = {acc:.3f}, "
                      f"FRR = {frr:.3f}, "
                      f"FAR = {far:.3f}, "
                      f"n_estimators: {param['n_estimators']}, "
                      f"max_samples: {param['max_samples']}, "
                      f"contamination: {param['contamination']}, "
                      f"max_features: {param['max_features']}, "
                      f"bootstrap: {param['bootstrap']}",
                      COLOR['None'])
            elif mode == 'ee':
                print(f">>> train.score: {train_score:.4f}, "
                      f"auc = {auc:.3f}, "
                      f"acc = {acc:.3f}, "
                      f"FRR = {frr:.3f}, "
                      f"FAR = {far:.3f}, "
                      f"store_precision: {param['store_precision']}, "
                      f"assume_centered: {param['assume_centered']}, "
                      f"support_fraction: {param['support_fraction']}, "
                      f"contamination: {param['contamination']}, ",
                      COLOR['None'])
            elif mode == 'locout':
                print(f">>> train.score: {best_param[1]:.4f}, "
                      f"auc = {auc:.4f}, "
                      f"acc = {acc:.4f}, "
                      f"FRR = {frr:.3f}, "
                      f"[FAR] = {far:.3f}, "
                      f"n_neighbors: {param['n_neighbors']}, "
                      f"algorithm: {param['algorithm']}, "
                      f"metric: {param['metric']}, "
                      f"contamination: {param['contamination']}, "
                      f"novelty: {param['novelty']}",
                      COLOR['None'], sep='')
            else:  # mode == 'svm'
                print(f">>> train.score: {train_score:.4f}, "
                      f"auc = {auc:.3f}, "
                      f"acc = {acc:.3f}, "
                      f"FRR = {frr:.3f}, "
                      f"FAR = {far:.3f}, "
                      f"nu: {param['nu']:.3f}, "
                      f"gamma: {param['gamma']}",
                      COLOR['None'])

    if mode == 'if':
        print(COLOR['red'],
              f">>> train.score: {best_param[1]:.4f}, "
              f"auc = {best_param[2]:.3f}, "
              f"acc = {best_param[3]:.3f}, "
              f"FRR = {best_param[4]:.3f}, "
              f"[FAR] = {best_param[5]:.3f}, "
              f"n_estimators: {best_param[6]}, "
              f"max_samples: {best_param[7]}, "
              f"contamination: {best_param[8]}, "
              f"max_features: {best_param[9]}, "
              f"bootstrap: {best_param[10]}, ",
              COLOR['None'], sep='')
    elif mode == 'ee':
        print(COLOR['red'],
              f">>> train.score: {best_param[1]:.4f}, "
              f"auc = {best_param[2]:.4f}, "
              f"acc = {best_param[3]:.4f}, "
              f"FRR = {best_param[4]:.3f}, "
              f"[FAR] = {best_param[5]:.3f}, "
              f"store_precision: {best_param[6]}, "
              f"assume_centered: {best_param[7]}, "
              f"support_fractionfloat: {best_param[8]}, "
              f"contamination: {best_param[9]}, ",
              COLOR['None'], sep='')
    elif mode == 'locout':
        print(COLOR['red'],
              f">>> train.score: {best_param[1]:.4f}, "
              f"auc = {best_param[2]:.4f}, "
              f"acc = {best_param[3]:.4f}, "
              f"FRR = {best_param[4]:.3f}, "
              f"[FAR] = {best_param[5]:.3f}, "
              f"n_neighbors: {best_param[6]}, "
              f"algorithm: {best_param[7]}, "
              f"metric: {best_param[8]}, "
              f"contamination: {best_param[9]}, "
              f"novelty: {best_param[10]}",
              COLOR['None'], sep='')
    else:  # mode == 'svm'
        print(COLOR['red'],
              f'>>> train.score: {best_param[1]:.4f}, '
              f"auc = {best_param[2]:.4f}, "
              f"acc = {best_param[3]:.4f}, "
              f"FRR = {best_param[4]:.3f}, "
              f"[FAR] = {best_param[5]:.3f}, "
              f"kernel = {best_param[6]}, "
              f"nu: {best_param[7]:.3f}, "
              f"gamma: {best_param[8]}",
              COLOR['None'], sep='')

    return best_param[0]  # best_model


def boost(user_name, user_features_path, test_sessions, mode=None):
    legal_features = load_features(user_features_path)
    illegal_features = get_all_illegal_test_features_for_user(user_name)
    illegal_features = shuffle(illegal_features)
    param_grid = ParameterGrid(param_grid=dict(
        learning_rate=[0.1, 1],
        n_estimators=[10, 25],
        max_depth=[1, 2, 3, 5],
        max_features=[1.0, 0.8, 0.5]
    ))
    y = np.ones(legal_features.shape[0] * 2)
    y[y.shape[0] // 2:] = -1
    best_param = (None, 1, 1, None, ...)  # (model, FRR, FAR, feature_importances, **param)
    for param in param_grid:
        GBM = GradientBoostingClassifier(**param)
        X = shuffle(illegal_features)[:legal_features.shape[0]]
        X = np.vstack((legal_features, X))
        _X, _y = shuffle(X, y)
        GBM.fit(_X, _y)
        _, _, frr, far = get_pairwise_score(GBM, user_name, test_sessions)
        if far < best_param[2]:
            best_param = (GBM, frr, far, GBM.feature_importances_, param)
            print(COLOR['yellow'], end='')
        print(f">>> FRR = {frr:.3f}, "
              f"FAR = {far:.3f}, "
              f"parameters: {param}",
              COLOR['None'])
    print(COLOR['red'],
          f">>> FRR = {best_param[1]:.3f}, "
          f"FAR = {best_param[2]:.3f}, "
          f"parameters: {best_param[4]}\n"
          f"feature_importances_: {np.argwhere(best_param[3]).squeeze()}",
          COLOR['None'], sep='')

    return best_param[0]


def get_users_model(session_dict, test_sessions, prefix='svm',
                    only_one_user=False, load_models=False, save_models=False):
    fit_model = classic_model
    if prefix == 'svm':
        print('> mode: OneClassSVM')
    elif prefix == 'if':
        print('> mode: IsolationForest')
    elif prefix == 'ee':
        print('> mode: EllipticEnvelope')
    elif prefix == 'locout':
        print('> mode: LocalOutlierFactor')
    elif prefix == 'boost':
        print('> mode: GradientBoostingClassifier')
        fit_model = boost
    else:
        print("[!] <mode> one of ['svm', 'if', 'ee', 'locout', 'boost']")
        exit(-1)

    users_model = dict()
    models_dir = f'../{prefix}_models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if load_models:
        print('> Model Loading...')
        for path in glob.glob(f'{models_dir}/*_{prefix}'):
            user = os.path.basename(path)[:6]
            with open(file=path, mode='rb') as f:
                users_model[user] = pickle.load(file=f)
        return users_model

    print('> Model.fit')
    svm_start_time = time.time()
    for user, user_session in session_dict.items():
        print(f'>> {user}')
        user_start_time = time.time()
        # vvvvvvvvvvvvvvv
        users_model[user] = fit_model(user, user_session, test_sessions, prefix)
        # ^^^^^^^^^^^^^^^
        print(f'>> {user} time: {time.time() - user_start_time:.3f} sec')
        if save_models:
            print('>> Model Saving...')
            with open(os.path.join(models_dir, user + f'_{prefix}'), mode='wb') as f:
                pickle.dump(obj=users_model[user], file=f)
        if only_one_user:
            print('>> [!] only_one_user')
            break

    print(f'> Model.fit time: {time.time() - svm_start_time:.3f} sec')
    return users_model


def get_all_mean_roc_curve(users_model, test_legal_sessions, mode='show'):
    print('> Model.mean_roc_curve')
    mean_mean_tpr = list()
    mean_auc, std_auc = list(), list()
    mean_frr, std_frr = list(), list()
    mean_far, std_far = list(), list()
    mean_fpr = np.linspace(0, 1, 100)
    for user, user_model in users_model.items():
        print(f'>> {user}')
        legal_features = get_legal_test_features_for_user(user)
        tprs, aucs = list(), list()
        frrs, fars = list(), list()
        fig, ax = plt.subplots(figsize=(8, 8))
        for other_user in test_legal_sessions:
            if other_user == user:
                continue
            other_user_features = get_legal_test_features_for_user(other_user)
            pairwise_features = np.vstack((legal_features, other_user_features))
            y_true = np.ones(pairwise_features.shape[0])
            y_true[-other_user_features.shape[0]:] = -1
            y_score = user_model.predict(pairwise_features)
            frrs.append(np.mean(y_score[:legal_features.shape[0]] == -1))
            fars.append(np.mean(y_score[-other_user_features.shape[0]:] == 1))
            y_score = user_model.decision_function(pairwise_features)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            ax.plot(fpr, tpr, label=f"ROC fold {other_user}", alpha=0.5)
            interp_tpr = np.interp(x=mean_fpr, xp=fpr, fp=tpr)
            interp_tpr[0] = 0.
            tprs.append(interp_tpr)
            aucs.append(auc(x=fpr, y=tpr))
        mean_far.append(np.mean(fars))
        std_far.append(np.std(fars))
        mean_frr.append(np.mean(frrs))
        std_frr.append(np.std(fars))

        ax.plot([0, 1], [0, 1], linestyle='--', color='r', alpha=0.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.
        mean_mean_tpr.append(mean_tpr)
        mean_auc.append(np.mean(aucs))
        std_auc.append(np.std(aucs))
        ax.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=1,
                label=f"Mean ROC (AUC = {np.mean(aucs):0.2f} \u00B1 {np.std(aucs):0.2f})")

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label='\u00B1 1 std. dev.')
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               xlabel='FPR', ylabel='TPR', title=f"ROC-curve {user}")
        ax.legend(loc="lower right")
        if mode == 'show':
            plt.show()
        elif mode == 'save':
            plt.savefig(f'../roc/roc_{user}.png', format='png', bbox_inches='tight')
        plt.close(fig)

    print('>> all')
    fig, ax = plt.subplots()
    mean_mean_tpr = np.mean(mean_mean_tpr, axis=0)
    ax.plot(mean_fpr, mean_mean_tpr, color='b', lw=2, alpha=1,
            label=f"Mean ROC (AUC = {np.mean(mean_auc):0.2f} \u00B1 {np.std(std_auc):0.2f})\n"
                  f"Mean FAR = {np.mean(mean_far):0.2f} \u00B1 {np.std(std_far):0.2f}\n"
                  f"Mean FRR = {np.mean(mean_frr):0.2f} \u00B1 {np.std(std_frr):0.2f}")
    ax.plot([0, 1], [0, 1], linestyle='--', color='r', alpha=0.5)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           xlabel='FPR', ylabel='TPR', title=f"ROC-curve")
    ax.legend(loc="lower right")
    if mode == 'show':
        plt.show()
    elif mode == 'save':
        plt.savefig(f'../roc/all_mean_roc.png', format='png', bbox_inches='tight')
    else:
        print("mode one of ['show', 'save']")
    plt.close(fig)


def print_score(users_model, test_legal_sessions):
    FRR, FAR = list(), list()
    for user, user_model in users_model.items():
        print(f'>> {user}')
        legal_features = get_legal_test_features_for_user(user)
        for other_user in test_legal_sessions:
            if other_user == user:
                continue
            other_user_features = get_legal_test_features_for_user(other_user)
            pairwise_features = np.vstack((legal_features, other_user_features))
            pairwise_features = SCALER.transform(pairwise_features)
            y_true = np.ones(pairwise_features.shape[0])
            y_true[-other_user_features.shape[0]:] = -1
            y_score = user_model.predict(pairwise_features)
            FRR.append(np.sum(y_score[:legal_features.shape[0]] == -1) / pairwise_features.shape[0])
            FAR.append(np.sum(y_score[-other_user_features.shape[0]:] == 1) / pairwise_features.shape[0])
    print(f"FRR = {np.mean(FRR) * 100:.1f} \u00B1 {np.std(FRR) * 100:.1f} %\n"
          f"FAR = {np.mean(FAR) * 100:.1f} \u00B1 {np.std(FAR) * 100:.1f} %")


if __name__ == '__main__':
    train_features_path = r'../features/n_features_81/5sec_5act/train_features'
    test_features_path = r'../features/n_features_81/5sec_5act/test_features'
    labels_path = r'../dataset/labels.csv'

    print('RUN')
    start_time = time.time()

    labels = pd.read_csv(labels_path)
    train_sessions = get_session_path(train_features_path)
    test_sessions = get_session_path(test_features_path, labels)
    test_legal_sessions, _ = split_into_legal(test_sessions, labels)
    load_all_test_features(test_legal_sessions)

    users_model = get_users_model(train_sessions, test_legal_sessions, prefix='svm',
                                  load_models=False, save_models=False)

    # get_all_mean_roc_curve(users_model, test_legal_sessions, mode='save')
    print_score(users_model, test_sessions)

    print(f'run time: {time.time() - start_time:.3f}')

#   ----------------------------------- DATASET 2 -----------------------------------
#     train_features_path = r'../features2/n_features_81/10sec_5act/train_features'
#
#     print('RUN')
#     start_time = time.time()
#     train_sessions = get_session_path(train_features_path)
#     test_sessions = train_sessions
#     load_all_test_features(test_sessions)
#
#     users_model = get_users_model(train_sessions, test_sessions, prefix='ee',
#                                   load_models=False, save_models=False, only_one_user=False)
#
#     # get_all_mean_roc_curve(users_model, test_sessions, mode='save')
#     print_score(users_model, test_sessions)
#     print(f'run time: {time.time() - start_time:.3f}')
