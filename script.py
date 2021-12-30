import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import os
import random
from snsynth.pytorch import PytorchDPSynthesizer
from snsynth.pytorch.nn import PATECTGAN
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def split_and_categorize(data):
    X = data.T[:-1]
    y = data.T[-1]
    return X.T, y.T

def run_knn(train_data, test_data):
    X_train, y_train = split_and_categorize(train_data)
    X_test, y_test = split_and_categorize(test_data)

    knn_clf = KNeighborsClassifier(10)
    knn_clf.fit(X_train, y_train)
    y_pred = knn_clf.predict(X_test)
    knn_cm = confusion_matrix(y_test, y_pred)
    return knn_cm

def run_mwem(train_data, T, eps, q_len, dim, starting_hist=None):
    #bins = [[i for i in range(1,9)], [i for i in range(1,9)], [i for i in range(1,16)], [i for i in range(1,8)], [i for i in range(1,7)], [i for i in range(1,4)], [i for i in range(1,4)]]
    bins = [[0,1,2]] * dim
    B, _ = np.histogramdd(train_data.to_numpy(), bins)
    B = B.flatten()
    D = B.shape
    Q = np.array([random_queries(D) for i in range(q_len)])

    A = better_mwem(B, D, Q, T, eps, part_avg=0.5, starting_hist=starting_hist)
    errors = [np.abs(run(q, B) - run(q, A)) / 30161 for q in Q]
    print(max(errors), np.mean(errors))

    A /= sum(A)
    syn_train = np.array(random.choices(np.arange(0, len(A)), weights=A, k=10000))

    headers = train_data.columns.to_list()
    bins.reverse()
    headers.reverse()
    X_syn = pd.DataFrame()

    for i in range(len(bins)):
        X_syn[headers[i]] = syn_train % (len(bins[i])-1)
        syn_train = np.floor(syn_train / (len(bins[i])-1))
    X_syn = X_syn.iloc[:, ::-1]
    X_syn[X_syn.columns] = X_syn[X_syn.columns].applymap(np.int64)
    return X_syn

def run_ann(train_data, test_data, dim, epochs=50, batch_size=1):
    X_train, y_train = split_and_categorize(train_data)
    X_test, y_test = split_and_categorize(test_data)

    classifier = Sequential()
    classifier.add(Dense(10, activation = 'relu', input_dim = dim-1))
    classifier.add(Dense(5, activation = 'relu'))
    classifier.add(Dense(1, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    history = classifier.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = batch_size, epochs = epochs, shuffle=True)

    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    ann_cm = confusion_matrix(y_test, y_pred)
    return ann_cm, history


def sample(sample_size, dim, class_vec):
    sample_low = np.random.choice([0, 1], (sample_size, dim-1))
    sample = np.zeros((sample_size, dim))
    for i in range(sample_size):
        sample[i][:-1] = sample_low[i]
        sample[i][-1] = 1 if np.sin(np.sum(np.bitwise_xor(sample[i][:-1].astype(int), class_vec))) >= 0 == 0 else 0
    return sample

def plot_tsne(data):
    embedding_1 = TSNE(n_components=2, perplexity=5.0, early_exaggeration=1.0).fit_transform(data[:1000])
    x,y = embedding_1.T

    plt.rcParams["figure.figsize"] = (15,15)
    plt.scatter(x,y,c=['blue' if data[i][-1] == 0 else 'red' for i in range(1000)])
    plt.title('t-Distributed Stochastic Neighbor Embedding of Data')
    plt.show()


def plot_history(history):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(history.history['loss'], label='train')
    ax[0].plot(history.history['val_loss'], label='test')
    ax[0].legend()
    ax[0].title.set_text("Loss")
    ax[1].plot(history.history['accuracy'], label='train')
    ax[1].plot(history.history['val_accuracy'], label='test')
    ax[1].legend()
    ax[1].title.set_text("Accuracy")
    plt.show()

dim = 12
train_size = 40
test_size = 60

class_vec = np.random.choice([0,1], dim-1)
train_data = sample(train_size, dim, class_vec)
test_data = sample(test_size, dim, class_vec)
plot_tsne(data)

extra_data = sample(1000, dim, class_vec)
plot_tsne(extra_data)

ann_cm, history = run_ann(train_data, test_data, dim, epochs=150)
plot_history(history)

gan = PytorchDPSynthesizer(10.0, PATECTGAN(regularization='dragan'), None)
gan.fit(df, categorical_columns=df.columns.to_list())
gan_syn = gan.sample(10000)

ann_cm, history = run_ann(gan_syn.to_numpy(), test_data, dim, epochs=150, batch_size=100)
plot_history(history)