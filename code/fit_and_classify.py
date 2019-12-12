from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import glob
import time
import os

""" https://dyakonov.org/2017/04/19/%D0%BF%D0%BE%D0%B8%D1%81%D0%BA-%D0%B0%D0%BD%D0%BE%D0%BC%D0%B0%D0%BB%D0%B8%D0%B9-anomaly-detection/ """

N_FEATURES = 60
# KEEP_LOG in Unix: python3 fit_and_classify.py | tee {LOG_FILE_NAME}

COLOR = {
    'yellow': '\033[93m',
    'green': '\033[92m',
    'cyan': '\033[96m',
    'red': '\033[91m',
    'None': '\033[0m'
}


def show_roc_curve(y_true, y_score, users=None, mode='show'):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(x=fpr, y=tpr)
    fig = plt.figure()
    plt.plot(fpr, tpr, color='r')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    if users is None:
        plt.title('ROC-curve')
    else:
        plt.title(f'ROC-curve ({users[0]} vs {users[1]})')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.text(0.7, 0.2, f'ROC-AUC={roc_auc:.3f}')
    if mode == 'show':
        plt.show()
    elif mode == 'save' and users is not None:
        plt.savefig(f'../roc/roc_{users[0]}_vs_{users[1]}.pdf', format='pdf')
    plt.close(fig=fig)


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
    # print(f'>>> features.shape={features.shape}')
    return features


def split_into_legal(test_session, labels):
    legal_session, illegal_session = {}, {}
    for user, user_sessions in test_session.items():
        for session in user_sessions:
            if labels[labels.filename == os.path.basename(session)].is_illegal.values:
                illegal_session.setdefault(user, []).append(session)
            else:
                legal_session.setdefault(user, []).append(session)

    return legal_session, illegal_session


def get_pairwise_auc(model, user_name, test_sessions):
    auc = list()
    legal_features = load_features(test_sessions[user_name])
    for other_user in test_sessions:
        if other_user == user_name:
            continue
        illegal_features = load_features(test_sessions[other_user])
        pairwise_features = np.vstack((legal_features, illegal_features))
        y_true = np.ones(pairwise_features.shape[0])
        y_true[legal_features.shape[0]:] = -1
        y_score = model.decision_function(pairwise_features)
        auc.append(roc_auc_score(y_true, y_score))
    return np.mean(auc)


def fit_model(user_name, user_features_path, test_sessions, mode):
    """ # https://scikit-learn.org/0.15/modules/generated/sklearn.svm.OneClassSVM.html """
    if mode == 'if':
        print('>>> mode: IsolationForest')
    elif mode == 'ee':
        print('>>> mode: EllipticEnvelope')
    else:
        print('>>> mode: OneClassSVM')
    features = load_features(user_features_path)
    if mode == 'if':
        param_grid = ParameterGrid(param_grid={
            'n_estimators': [100],
            'max_samples': [1.0, 0.9],
            'contamination': ['auto', 0.05],
            'max_features': [1.0, 0.9],
            'bootstrap': [True, False],
            'behaviour': ['deprecated']
        })
    elif mode == 'ee':
        param_grid = ParameterGrid(param_grid={
            'store_precision': [True, False],
            'assume_centered': [False],
            'support_fraction': [1.],
            'contamination': [0.1, 0.05],
        })
    else:  # mode == 'svm'
        param_grid = ParameterGrid(param_grid={
            'kernel': ['rbf'],
            'nu': [0.10, 0.05],  # np.linspace(0.3, 0.05, 6),
            'gamma': [1e-5, 1e-4],
        })
    best_param = (None, 0, 0, ...)  # (model, train_score, test_auc, **param_grid)
    for param in param_grid:
        if mode == 'if':
            model = IsolationForest(**param)
        elif mode == 'ee':
            model = EllipticEnvelope(**param)
        else:  # mode == 'svm'
            model = OneClassSVM(**param)
        model.fit(X=features)
        train_score = np.mean(model.predict(features) == 1)
        auc = get_pairwise_auc(model, user_name, test_sessions)
        if auc > best_param[2]:
            if mode == 'svm':
                best_param = (model, train_score, auc, param['kernel'], param['nu'], param['gamma'])
            elif mode == 'if':
                best_param = (model, train_score, auc, param['n_estimators'], param['max_samples'],
                              param['contamination'], param['max_features'], param['bootstrap'])
            elif mode == 'ee':
                best_param = (model, train_score, auc, param['store_precision'], param['assume_centered'],
                              param['support_fraction'], param['contamination'])
            print(COLOR['yellow'], end='')

        if mode == 'if':
            print(f">>> train.score: {train_score:.4f}, "
                  f"auc = {auc:.4f}, "
                  f"n_estimators: {param['n_estimators']}, "
                  f"max_samples: {param['max_samples']}, "
                  f"contamination: {param['contamination']}, "
                  f"max_features: {param['max_features']}, "
                  f"bootstrap: {param['bootstrap']}",
                  COLOR['None'])
        elif mode == 'ee':
            print(f">>> train.score: {train_score:.4f}, "
                  f"auc = {auc:.4f}, "
                  f"store_precision: {param['store_precision']}, "
                  f"assume_centered: {param['assume_centered']}, "
                  f"support_fraction: {param['support_fraction']}, "
                  f"contamination: {param['contamination']}, ",
                  COLOR['None'])
        else:  # mode == 'svm'
            print(f">>> train.score: {train_score:.4f}, "
                  f"auc = {auc:.4f}, "
                  f"nu: {param['nu']:.2f}, "
                  f"gamma: {param['gamma']}",
                  COLOR['None'])

    if mode == 'if':
        print(COLOR['red'],
              f">>> train.score: {best_param[1]:.4f}, "
              f"auc = {best_param[2]:.4f}, "
              f"n_estimators: {best_param[3]}, "
              f"max_samples: {best_param[4]}, "
              f"contamination: {best_param[5]}, "
              f"max_features: {best_param[6]}, "
              f"bootstrap: {best_param[7]}, ",
              COLOR['None'], sep='')
    elif mode == 'ee':
        print(COLOR['red'],
              f">>> train.score: {best_param[1]:.4f}, "
              f"auc = {best_param[2]:.4f}, "
              f"store_precision: {best_param[3]}, "
              f"assume_centered: {best_param[4]}, "
              f"support_fractionfloat: {best_param[5]}, "
              f"contamination: {best_param[6]}, ",
              COLOR['None'], sep='')
    else:  # mode == 'svm'
        print(COLOR['red'],
              f'>>> train.score: {best_param[1]:.4f}, '
              f"auc = {best_param[2]:.4f}, "
              f"nu: {best_param[4]:.2f}, "
              f"gamma: {best_param[5]}",
              COLOR['None'], sep='')

    return best_param[0]  # best_model


def get_users_model(session_dict, test_sessions, prefix='svm',
                    only_one_user=False, load_models=False, save_models=False):
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
        # vvvvvvvvvvvvv
        users_model[user] = fit_model(user, user_session, test_sessions, prefix)
        # ^^^^^^^^^^^^^
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


def classify(users_model, test_legal_sessions, only_one_user=False):
    print('> Model.classify')
    svm_start_time = time.time()
    for user, user_model in users_model.items():
        print(f'>> {user}')
        user_start_time = time.time()
        legal_features = load_features(test_legal_sessions[user])
        for other_user in test_legal_sessions:
            if other_user == user:
                continue
            other_user_features = load_features(test_legal_sessions[other_user])
            pairwise_features = np.vstack((legal_features, other_user_features))
            label = np.ones(pairwise_features.shape[0])
            label[-other_user_features.shape[0]:] = -1
            score = user_model.decision_function(pairwise_features)
            show_roc_curve(label, score, users=[user[-2:], other_user[-2:]], mode='show')

        print(f'>> {user} time: {time.time() - user_start_time:.3f} sec')
        if only_one_user:
            print('>> [!] only_one_user')
            break
    print(f'> Model.predict time: {time.time() - svm_start_time:.3f} sec')


def get_all_mean_roc_curve(users_model, test_legal_sessions, mode='show'):
    from scipy import interp

    print('> Model.mean_roc_curve')
    mean_mean_tpr = list()
    mean_auc, std_auc = list(), list()
    mean_fpr = np.linspace(0, 1, 100)
    for user, user_model in users_model.items():
        print(f'>> {user}')
        legal_features = load_features(test_legal_sessions[user])
        tprs, aucs = list(), list()
        fig, ax = plt.subplots(figsize=(8, 8))
        for other_user in test_legal_sessions:
            if other_user == user:
                continue
            other_user_features = load_features(test_legal_sessions[other_user])
            pairwise_features = np.vstack((legal_features, other_user_features))
            y_true = np.ones(pairwise_features.shape[0])
            y_true[-other_user_features.shape[0]:] = -1
            y_score = user_model.decision_function(pairwise_features)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            ax.plot(fpr, tpr, label=f"ROC fold {other_user}", alpha=0.5)
            interp_tpr = interp(x=mean_fpr, xp=fpr, fp=tpr)
            interp_tpr[0] = 0.
            tprs.append(interp_tpr)
            aucs.append(auc(x=fpr, y=tpr))

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
            label=f"Mean ROC (AUC = {np.mean(mean_auc):0.2f} \u00B1 {np.std(std_auc):0.2f})")
    ax.plot([0, 1], [0, 1], linestyle='--', color='r', alpha=0.8)
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


def get_all_mean_pr_curve(users_model, test_legal_sessions, mode='show'):
    from scipy import interp

    print('> Model.mean_pr_curve')
    for user, user_model in users_model.items():
        print(f'>> {user}')
        legal_features = load_features(test_legal_sessions[user])
        tprs, aucs = list(), list()
        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots(figsize=(8, 8))
        for other_user in test_legal_sessions:
            if other_user == user:
                continue
            other_user_features = load_features(test_legal_sessions[other_user])
            pairwise_features = np.vstack((legal_features, other_user_features))
            y_true = np.ones(pairwise_features.shape[0])
            y_true[-other_user_features.shape[0]:] = -1
            y_score = user_model.decision_function(pairwise_features)
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ax.plot(recall, precision, label=f"PR fold {other_user}", alpha=0.5)
            interp_tpr = interp(x=mean_fpr, xp=recall, fp=precision)
            interp_tpr[0] = 1.
            tprs.append(interp_tpr)
            aucs.append(auc(x=recall, y=precision))

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 0.
        ax.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=1,
                label=f"Mean PR (AUC = {np.mean(aucs):0.2f} \u00B1 {np.std(aucs):0.2f})")

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label='\u00B1 1 std. dev.')
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               xlabel='Recall', ylabel='Precision', title=f"PR-curve {user}")
        ax.legend(loc="lower left")
        if mode == 'show':
            plt.show()
        elif mode == 'save':
            plt.savefig(f'../pr/pr_{user}.png', format='png', bbox_inches='tight')
        else:
            print("mode one of ['show', 'save']")
        plt.close(fig)


if __name__ == '__main__':
    train_features_path = r'../features/train_features'
    test_features_path = r'../features/test_features'
    labels_path = r'../dataset/labels.csv'

    print('RUN')
    start_time = time.time()

    labels = pd.read_csv(labels_path)
    train_sessions = get_session_path(train_features_path)
    test_sessions = get_session_path(test_features_path, labels)
    test_legal_sessions, _ = split_into_legal(test_sessions, labels)

    # fit:
    users_model = get_users_model(train_sessions, test_legal_sessions, prefix='ee',
                                  load_models=False, save_models=True)
    # load:
    # users_model = get_users_model(train_sessions, test_legal_sessions, prefix='svm',
    #                               load_models=True, save_models=False)
    get_all_mean_roc_curve(users_model, test_legal_sessions, mode='save')
    # get_all_mean_pr_curve(users_model, test_legal_sessions, mode='show')
    # classify(users_model, test_legal_sessions, only_one_user=False)

    print(f'run time: {time.time() - start_time:.3f}')

"""################################### DATASET 2 ###################################"""
# train_features_path = r'../features2/train_features'
# test_features_path = r'../features2/test_features'
#
# print('RUN')
# start_time = time.time()
# train_sessions = get_session_path(train_features_path)
# test_sessions = get_session_path(test_features_path)
#
# # fit:
# users_model = get_users_model(train_sessions, test_sessions, prefix='svm',
#                               load_models=False, save_models=True, only_one_user=True)
# # load:
# # users_model = get_users_model(train_sessions, test_legal_sessions, prefix='svm',
# #                               load_models=True, save_models=False)
#
# get_all_mean_roc_curve(users_model, test_sessions, mode='show')
# print(f'run time: {time.time() - start_time:.3f}')
