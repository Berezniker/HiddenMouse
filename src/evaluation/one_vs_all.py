from classification.neural_network import NeuralNetwork
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from utils import describe_data
import utils.constants as const
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import glob
import time
import os


def EER(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Calculate EER metrics

    :param y_true: true labels
    :param y_score: target scores
    :return: EER
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(f=lambda x: 1.0 - x - interp1d(fpr, tpr)(x), a=0.0, b=1.0)
    return eer


def AUC(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Calculate AUC metrics

    :param y_true: true labels
    :param y_score: target scores
    :return: ROC AUC
    """
    return roc_auc_score(y_true, y_score)


def get_data(file_path: str, preprocessor=None) -> np.ndarray:
    """
    Loading data

    :param file_path: file path
    :param preprocessor: preprocessing function
    :return: data
    """
    db = pd.read_csv(file_path, header=None)
    db.dropna(inplace=True)
    data = db.values
    if preprocessor is not None:
        data = preprocessor.transform(data)
    return data


def one_vs_all(classifier, parameters: dict) -> dict:
    """
    Evaluation of the classifier quality

    :param classifier: base classifier
    :param parameters: classifier constructor parameters
    :return: dictionary like: {LEGAL_DATASET: {USER_NAME: {ILLEGAL_DATASET: AUC_list, ...} } }
    """
    res = dict()
    for legal_dataset in const.ALL_DATASET_NAME:
        print(f"{legal_dataset}")
        res[legal_dataset] = dict()
        legal_path = os.path.join(const.DATASET_PATH,
                                  f"{legal_dataset}/train_features/user*")

        for path in glob.glob(legal_path):
            legal_user_name = os.path.basename(path)
            res[legal_dataset][legal_user_name] = dict()
            print(legal_user_name, end=' $ ')
            scaler = StandardScaler()
            data_train_path = os.path.join(path, const.COMBINE_SESSION_NAME)
            X_train = scaler.fit_transform(get_data(data_train_path))
            # ---------------------------------------- #
            clf = classifier(**parameters).fit(X_train)
            # ---------------------------------------- #
            data_valid_path = data_train_path.replace("train", "test")
            X_valid = get_data(data_valid_path, scaler)
            valid_decision = clf.decision_function(X_valid)

            for illegal_dataset in const.ALL_DATASET_NAME:
                print(f"... vs {illegal_dataset}", end=" ")
                illegal_path = os.path.join(const.DATASET_PATH,
                                            f"{illegal_dataset}/test_features/user*")
                res[legal_dataset][legal_user_name][illegal_dataset] = list()
                score_list = list()
                for i_path in glob.glob(illegal_path):
                    if legal_dataset == illegal_dataset:
                        if legal_user_name == os.path.basename(i_path):
                            continue
                    X_test = get_data(os.path.join(i_path, const.COMBINE_SESSION_NAME), scaler)
                    y = np.array([1] * len(X_valid) + [-1] * len(X_test))
                    y_score = np.hstack((valid_decision, clf.decision_function(X_test)))
                    score_list.append(AUC(y, y_score))

                res[legal_dataset][legal_user_name][illegal_dataset] = np.array(score_list)
            print()

    return res


def boxplot(res: dict, path: str) -> None:
    """
    Visualization of the classifier quality by BoxPlot

    :param res: output dictionary of function one_vs_all
    :param path: global save path
    :return: None
    """
    for legal_dataset in res:
        dir_path = os.path.join(path, legal_dataset)
        os.makedirs(dir_path, exist_ok=True)
        all_score = list()
        for legal_user_name in res[legal_dataset]:
            print(f"{legal_user_name}", end=' $ ')
            data = res[legal_dataset][legal_user_name]
            color = [f'hsl({str(h)}%, 50%, 50%)' for h in np.linspace(0, 360, len(data))]

            fig = go.Figure(
                data=[go.Box(y=y, marker_color=color[i], name=illegal_dataset)
                      for i, (illegal_dataset, y) in enumerate(data.items())]
            )

            mean_AUC = np.array([v.mean() for v in data.values()]).mean()
            all_score.append(mean_AUC)
            fig.update_layout(
                title=f"{legal_dataset}.{legal_user_name} vs ALL: mean AUC = {mean_AUC:.2f}",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(title="AUC", zeroline=False, gridcolor='white')
            )
            # fig.show()
            save_path = os.path.join(dir_path, f"{legal_user_name}_mean-AUC={mean_AUC:.2f}.png")
            fig.write_image(save_path)

        with open(os.path.join(dir_path, "mean_auc.txt"), mode='w') as f:
            f.write(f"mean AUC = {np.mean(all_score):.3f}")

        print(f"\nmean AUC = {np.mean(all_score):.3f}\n")


def save_log(data: dict, classifier_name: str, parameters: dict, comment: str = "") -> dict:
    """
    Adds information about the classifier and metric to the json log file

    :param data: output dictionary of function one_vs_all
    :param classifier_name: base classifier name
    :param parameters: classifier constructor parameters
    :param comment: experiment comment
    :return: dictionary describing data in json format
    """
    describer = describe_data.load_log()

    for dataset in data:
        dataset_mean_AUC = list()
        for user in data[dataset]:
            mean_AUC = np.mean([auc.mean() for auc in data[dataset][user].values()])
            dataset_mean_AUC.append(mean_AUC)
            describer[dataset][user].setdefault('classifier', list())
            describer[dataset][user]['classifier'].append({
                "name": classifier_name,
                "parameters": parameters,
                "mean_AUC": np.mean(dataset_mean_AUC).round(3)
            })
        describer[dataset].setdefault('score', list())
        describer[dataset]['score'].append({
            "classifier": classifier_name,
            "comment": comment,
            "parameters": parameters,
            "mean_AUC": np.mean(dataset_mean_AUC).round(3)
        })

    describe_data.dump_log(describer)

    return describer


if __name__ == "__main__":
    # classifier = OneClassSVM
    # parameters = {"kernel": 'rbf', "gamma": 'scale', "nu": 0.05}

    classifier = NeuralNetwork
    parameters = {"mode": const.NN.AUTOENCODER, "n_features": const.N_FEATURES,
                  "optimizer": 'Adam', "loss": 'logcosh'}

    # classifier = NeuralNetwork
    # parameters = {"mode": const.NN.CNN, "n_features": const.N_FEATURES, "optimizer": 'Adam', "loss": 'logcosh'}

    print("Run!")
    start_time = time.time()
    pairwise_data = one_vs_all(classifier, parameters)
    # save_log(pairwise_data, classifier.__name__, parameters)
    # boxplot(pairwise_data, f"/score/{classifier.__name__}_nu={parameters['nu']}")
    boxplot(pairwise_data, f"/score/{classifier.__name__}")
    print(f"End of run. Time: {(time.time() - start_time) / 60.0:.1f} min")
