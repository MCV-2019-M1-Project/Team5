import ml_metrics as metrics
from definitions import K


def evaluateMAP(actual, predicted):
    return metrics.mapk(actual, predicted, K)


def evaluateAP(actual, predicted):
    return metrics.apk(actual, predicted, K)
