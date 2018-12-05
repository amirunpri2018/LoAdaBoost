from keras.models import Model, load_model
from keras import layers
from keras.layers import Input
import numpy as np


def average(models):
    weights_arrays = []
    for model in models:
        weights = model.get_weights()
        weights_arrays.append(weights)
    average_weights = np.average(weights_arrays, 0)
    return average_weights


def weighted_average(models, aucs):
    weights_arrays = []

    for model in models:
        weights = model.get_weights()
        weights_arrays.append(weights)
    aucs = np.array(aucs)
    norm_aucs = aucs / np.float(aucs.sum())

    for i in range(len(weights_arrays)):
        weights_array = weights_arrays[i]
        for j in range(8):
            weights_array[j] = weights_array[j] * norm_aucs[i]
    weights = np.sum(weights_arrays, axis=0)

    return weights


def exp_weighted_average(models, aucs):
    weights_arrays = []

    for model in models:
        weights = model.get_weights()
        weights_arrays.append(weights)
    aucs = np.exp(np.array(aucs))
    norm_aucs = aucs / np.float(aucs.sum())

    for i in range(len(weights_arrays)):
        weights_array = weights_arrays[i]
        for j in range(8):
            weights_array[j] = weights_array[j] * norm_aucs[i]
    weights = np.sum(weights_arrays, axis=0)

    return weights


def reversed_weighted_average(models, aucs):
    weights_arrays = []

    for model in models:
        weights = model.get_weights()
        weights_arrays.append(weights)
    aucs = np.array(aucs)
    norm_aucs = aucs / np.float(aucs.sum())
    norm_aucs = 1 - norm_aucs
    norm_aucs = norm_aucs / np.float(norm_aucs.sum())

    for i in range(len(weights_arrays)):
        weights_array = weights_arrays[i]
        for j in range(8):
            weights_array[j] = weights_array[j] * norm_aucs[i]
    weights = np.sum(weights_arrays, axis=0)

    return weights
