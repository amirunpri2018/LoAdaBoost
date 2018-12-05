import pandas as pd
import numpy as np
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from model_average import *
import matplotlib.pyplot as plt
from keras import initializers


def ann(X_train, Y_train, random_seed, batch_size_specified=100, dropout_rate=0.5):
    #model

    input_shape = X_train.shape[1]
    # input layer
    input_layer = Input(shape=(input_shape,))
    # hidden layers
    hidden_layer1 = Dense(20, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=random_seed))(input_layer)
    hidden_layer1 = Dropout(dropout_rate)(hidden_layer1)
    hidden_layer2 = Dense(10, activation='relu')(hidden_layer1)
    hidden_layer2 = Dropout(dropout_rate)(hidden_layer2)
    hidden_layer3 = Dense(5, activation='relu')(hidden_layer2)
    hidden_layer3 = Dropout(dropout_rate)(hidden_layer3)
    # output layer
    output_layer = Dense(1, activation='sigmoid')(hidden_layer3)

    ann_model = Model(inputs=input_layer, outputs=output_layer)

    ann_model.compile(optimizer='adam', loss='binary_crossentropy')

    #ann_model.summary()

    history = ann_model.fit(X_train, Y_train,
                    epochs=5,
                    batch_size=batch_size_specified,
                    shuffle=True,
                    verbose=False)
    loss = history.history["loss"][-1]
    return ann_model, loss


def ann2(X_train, Y_train, initializers, batch_size_specified=100, dropout_rate=0.5):
    kernel_indices = [0, 2, 4, 6]
    bias_indices = [1, 3, 5, 7]
    kernel_initializers = np.array(initializers)[kernel_indices]
    bias_initializers = np.array(initializers)[bias_indices]

    #model

    input_shape = X_train.shape[1]
    # input layer
    input_layer = Input(shape=(input_shape,))
    # hidden layers
    hidden_layer1 = Dense(20, activation='relu', weights=[kernel_initializers[0], bias_initializers[0]])(input_layer)
    hidden_layer1 = Dropout(dropout_rate)(hidden_layer1)
    hidden_layer2 = Dense(10, activation='relu', weights=[kernel_initializers[1], bias_initializers[1]])(hidden_layer1)
    hidden_layer2 = Dropout(dropout_rate)(hidden_layer2)
    hidden_layer3 = Dense(5, activation='relu', weights=[kernel_initializers[2], bias_initializers[2]])(hidden_layer2)
    hidden_layer3 = Dropout(dropout_rate)(hidden_layer3)
    # output layer
    output_layer = Dense(1, activation='sigmoid', weights=[kernel_initializers[3], bias_initializers[3]])(hidden_layer3)

    ann_model = Model(inputs=input_layer, outputs=output_layer)

    ann_model.compile(optimizer='adam', loss='binary_crossentropy')

    #ann_model.summary()

    history = ann_model.fit(X_train, Y_train,
                    epochs=5,
                    batch_size=batch_size_specified,
                    shuffle=True,
                    verbose=False)
    loss = history.history["loss"][-1]
    return ann_model, loss


def calculate_auc(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def federated_learning(num_of_clients):

    average_training_aucs = []
    end_of_loop_test_aucs = []

    # global loops
    for t in range(30):

        np.random.seed(t+1)
        indices =  np.random.choice(100, num_of_clients, replace=False)

        #print indices

        print "round " + str(t+1) +" start, random seed=" + str(t+1)

        X_train_clients = X_train_100_shares[indices]
        Y_train_clients = Y_train_100_shares[indices]

        anns = []
        roc_aucs = []
        test_aucs = []
        losses = []
        for i in range(num_of_clients):

            if t == 0:
                ann_model, loss = ann(np.array(X_train_clients[i]), np.array(Y_train_clients[i]), random_seed=t+1,
                                      batch_size_specified=30,
                                      dropout_rate=0.0)
            else:
                ann_model, loss = ann2(np.array(X_train_clients[i]), np.array(Y_train_clients[i]),
                                       initializers=weights, batch_size_specified=30, dropout_rate=0.0)

            anns.append(ann_model)
            # calculate auc for model trained with each client
            roc_auc = calculate_auc(ann_model, np.array(X_train_clients[i]), np.array(Y_train_clients[i]))
            roc_aucs.append(roc_auc)

            #loss
            losses.append(loss)

            #test auc
            test_auc = calculate_auc(ann_model, X_test, Y_test)
            test_aucs.append(test_auc)

            #print "round " + str(t+2) + "  client " + str(i+1) + "  loss=" + str(loss) +\
            #      "  training auc=" + str(roc_auc) + "  test auc=" + str(test_auc)

        average_training_auc = np.average(roc_aucs)
        print "round " + str(t+1) + " average training auc=" + str(average_training_auc)
        average_training_aucs.append(average_training_auc)

        anns[0].set_weights(average(anns))

        end_of_loop_test_auc = calculate_auc(anns[0], X_test, Y_test)
        print "round " + str(t+1) + " test auc=" + str(end_of_loop_test_auc)
        end_of_loop_test_aucs.append(end_of_loop_test_auc)

        weights = anns[0].get_weights()

    federated_ann = anns[0]
    federated_ann.summary()
    return federated_ann, average_training_aucs, end_of_loop_test_aucs


def prepare_data():
    X_train = pd.read_csv("./IID_data/X_train.csv", dtype="int", header=None).values
    Y_train = pd.read_csv("./IID_data/Y_train.csv", dtype="int", header=None).values

    X_train_100_shares = np.array(np.array_split(X_train, 100))
    Y_train_100_shares = np.array(np.array_split(Y_train, 100))

    X_test = pd.read_csv("./IID_data/X_test.csv", dtype="int", header=None).values
    Y_test = pd.read_csv("./IID_data/Y_test.csv", dtype="int", header=None).values
    return X_train_100_shares, Y_train_100_shares, X_test, Y_test


def federated_learning_evaluation(num_of_clients):
    print "CLIENTS " + str(num_of_clients) + "% START"
    federated_ann, average_training_aucs, end_of_loop_test_aucs = federated_learning(num_of_clients)
    #pd.DataFrame(average_training_aucs).to_csv("./IID_evaluation/"+ str(num_of_clients) + "simple_average_training_aucs.csv",
    #                                           header=False, index=False)
    pd.DataFrame(end_of_loop_test_aucs).to_csv("./IID_evaluation/"+ str(num_of_clients) + "average_test_aucs.csv",
                                               header=False, index=False)
    print "CLIENTS " + str(num_of_clients) + "% END"

X_train_100_shares, Y_train_100_shares, X_test, Y_test = prepare_data()

#federated_learning_evaluation(10)

federated_learning_evaluation(20)

federated_learning_evaluation(50)

federated_learning_evaluation(90)
