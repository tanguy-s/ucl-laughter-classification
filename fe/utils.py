import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_data(balance_ones=True):
    # images are 48x48 = 2304 size vectors
    # N = 35887
    Y = []
    X = []
    first = True
    for line in open('fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)

    if balance_ones:
        # balance the 1 class
        X0, Y0 = X[Y!=1, :], Y[Y!=1]
        X1 = X[Y==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1]*len(X1)))

    return X, Y



def get_image_data():
    print('Retrieving Data')
    X, Y = get_data()
    print('Data Retrieved')
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y





def saveModel(sess,model_file):
    print("Saving model at: " + model_file)
    saver = tf.train.Saver()
    saver.save(sess, model_file)
    print("Saved")


def loadModel(sess, model_file):

    print("Loading model from: " + model_file)
    # load saved session
    loader = tf.train.Saver()
    loader.restore(sess, model_file)
    print("Loaded")


def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind