from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
import numpy as np
from data_splits import DataSplits


def average_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    avg_accuracy = np.mean(per_class_accuracy)
    return avg_accuracy


def evaluate_train_test_pair(evaluation_train_x, evaluation_train_y, evaluation_test_x, evaluation_test_y):
    evaluator_algorithm = get_metric_evaluator()
    evaluator_algorithm.fit(evaluation_train_x, evaluation_train_y)
    y_pred = evaluator_algorithm.predict(evaluation_test_x)
    return calculate_metrics(evaluation_test_y, y_pred)


def evaluate_split(split:DataSplits, transform=None):
    if transform is None:
        evaluation_train_x = split.evaluation_train_x
        evaluation_test_x = split.evaluation_test_x
    else:
        evaluation_train_x = transform.transform(split.evaluation_train_x)
        evaluation_test_x = transform.transform(split.evaluation_test_x)
    return evaluate_train_test_pair(evaluation_train_x, split.evaluation_train_y, evaluation_test_x, split.evaluation_test_y)


def calculate_metrics(y_test, y_pred):
    oa = accuracy_score(y_test, y_pred)
    aa = average_accuracy(y_test, y_pred)
    k = cohen_kappa_score(y_test, y_pred)
    return oa, aa, k


def get_metric_evaluator():
    gowith = "sv"

    if gowith == "rf":
        return RandomForestClassifier()
    elif gowith == "sv":
        return SVC(C=1e5, kernel='rbf', gamma=1.)
    else:
        return MLPClassifier(max_iter=2000)