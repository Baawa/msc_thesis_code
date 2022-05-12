import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def get_binary_classification_performance(y_hat, y_true):
    acc = accuracy_score(y_true, y_hat)
    f1_class_0 = f1_score(y_true, y_hat, pos_label=0)
    f1_class_1 = f1_score(y_true, y_hat, pos_label=1)

    return acc, f1_class_0, f1_class_1


def get_multiclass_classification_performance(y_hat, y_true):
    acc = accuracy_score(y_true, y_hat)
    macro_f1_score = f1_score(y_true, y_hat, average="macro")

    return acc, macro_f1_score


def get_coverage_and_efficiency(confidence_intervals, y_true):
    coverage = np.sum([1 if (y_true[i] in confidence_intervals[i]) else 0
                       for i in range(len(confidence_intervals))])/len(y_true)

    confidence_interval_sizes = np.asarray([len(confidence_intervals[i]) for i in range(len(confidence_intervals))])
    avg_prediction_set_size = confidence_interval_sizes.mean()

    # % of singleton predictions
    num_singleton_pred = confidence_interval_sizes[confidence_interval_sizes == 1].shape[0]
    num_pred = confidence_interval_sizes.shape[0]
    frac_singleton_pred = num_singleton_pred / num_pred

    num_empty_pred = num_singleton_pred = confidence_interval_sizes[confidence_interval_sizes == 0].shape[0]
    frac_empty_pred = num_empty_pred / num_pred

    return coverage, avg_prediction_set_size, frac_singleton_pred, frac_empty_pred
