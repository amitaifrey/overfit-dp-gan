import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import os
import numba
from numba import jit, prange

# Generate random coefficients in [-1,1] for a linear query, where a third of them are expected to be 0.
# This was done to simulate a linear query that will probably not reach counting over 70% of the dataset.
# I tested this with other parameters as well, such as coefficients in [0,1], indicator variables, no zeroing, 
# allowing only 20% to have values and others. I got similar results in all of the runs.
def random_queries(shape):
    return np.random.random(shape) *  np.random.choice([-1,0,1], size=shape)

@numba.jit(cache=True, nopython=True)
def run(q, hist):
    return np.sum(hist * q)

@numba.jit(cache=True, nopython=True, parallel=True)
def mult_weights(A, q, diff, size):
    for i in prange(len(A)):
        # Notice we can simply multiply q[i]*A[i] because the rest of x (the record) is 0 so no need to multiply it at all
        A[i] *= np.exp(q[i] * A[i] * diff / (2.0 * size)) 
    return A * size / np.sum(A)

@numba.jit(cache=True, nopython=True, parallel=True)
def calc_scores(Q, A, B, eps):
    scores = np.zeros(len(Q))
    for i in prange(len(Q)):
        scores[i] = np.exp(eps * np.abs(run(Q[i], A) - run(Q[i], B)) / 2.0)
    return scores / np.linalg.norm(scores, ord=1)

def better_mwem(B, D, Q, T, eps, part_avg=0):
    # initialize A
    A = [np.full(B.shape, np.sum(B) / B.size)]

    for i in range(1, T+1):
        if i % 10 == 0:
            print(i)
        curr_eps = eps/(2.0*i)

        # choose query
        scores = calc_scores(Q, A[i-1], B, curr_eps)
        q_idx = np.random.choice(np.arange(0,len(Q)), p=scores)
        q = Q[q_idx]
        
        # run it
        actual = run(q, B)
        noisy = actual + np.random.laplace(0, 1/curr_eps)

        # update weights
        A_i = A[i-1]
        old = run(q, A_i)
        A_i = mult_weights(A_i, q, noisy - old, np.sum(B))
        A.append(A_i)
    return np.average(A[int(len(A)*part_avg):], axis=0)

@numba.jit(cache=True, nopython=True)
def sample_index(vec):
    for i in range(len(vec)):
        if np.random.rand() <= vec[i]:
            return i
    return np.random.randint(1,len(vec))

@numba.jit(cache=True, nopython=True, parallel=True)
def sample_syn_data(dist, size):
    syn_data = np.zeros(size)
    for i in prange(size):
        syn_data[i] = sample_index(dist)
    return syn_data

def process_data(file):
    data = pd.read_csv(file)
    data = data.drop("fnlwgt",1)
    data = data.drop("capital-gain",1)
    data = data.drop("capital-loss",1)
    data = data.drop("age",1)
    data = data.drop("hours-per-week",1)
    data = data.drop("native-country",1)
    data = data.drop("education",1)
    data = data.drop("education-num",1)

    data.replace(["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"], [1,2,3,4,5,6,7,8], inplace=True)
    data.replace(["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"], [1,2,3,4,5,6,7], inplace=True)
    data.replace(["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"], [1,2,3,4,5,6,7,8,9,10,11,12,13,14], inplace=True)
    data.replace(["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"], [1,2,3,4,5,6], inplace=True)
    data.replace(["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"], [1,2,3,4,5], inplace=True)
    data.replace(["Female", "Male"], [1,2], inplace=True)
    data.replace(["<=50K", ">50K"], [0,1], inplace=True)
    #data.replace(["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], inplace=True)
    #data.replace(["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong"], [i for i in range(1,41)], inplace=True)

    return data

def split_and_categorize(data):
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    numeric_transformer = Pipeline(steps=[('func', FunctionTransformer(lambda x: x))])

    numeric_features = []
    categorical_features = ["workclass", "marital-status", "occupation", "relationship", "race", "sex"]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    X = data[data.columns[:-1]]
    y = data[data.columns[-1]]

    return preprocessor.fit_transform(X).toarray().astype(int), LabelEncoder().fit_transform(y)

def run_ann(train_data, test_data, epochs=50):
    X_train, y_train = split_and_categorize(train_data)
    X_test, y_test = split_and_categorize(test_data)

    classifier = Sequential()
    classifier.add(Dense(24, activation = 'relu', input_dim = 41))
    classifier.add(Dense(12, activation = 'relu'))
    classifier.add(Dense(6, activation = 'relu'))
    classifier.add(Dense(1, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = 10, epochs = epochs, shuffle=True)

    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    ann_cm = confusion_matrix(y_test, y_pred)
    return ann_cm

def run_dt(train_data, test_data):
    X_train, y_train = split_and_categorize(train_data)
    X_test, y_test = split_and_categorize(test_data)

    dt_clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=5, min_samples_leaf=5)
    dt_clf_gini.fit(X_train, y_train)
    y_pred_gini = dt_clf_gini.predict(X_test)
    dt_cm = confusion_matrix(y_test, y_pred_gini)
    return dt_cm

def run_knn(train_data, test_data):
    X_train, y_train = split_and_categorize(train_data)
    X_test, y_test = split_and_categorize(test_data)

    knn_clf = KNeighborsClassifier(2)
    knn_clf.fit(X_train, y_train)
    y_pred = knn_clf.predict(X_test)
    knn_cm = confusion_matrix(y_test, y_pred)
    return knn_cm

def run_mwem(train_data, T, eps, q_len):
    bins = [[i for i in range(1,9)], [i for i in range(1,9)], [i for i in range(1,16)], [i for i in range(1,8)], [i for i in range(1,7)], [i for i in range(1,4)], [i for i in range(1,4)]]
    B, _ = np.histogramdd(train_data.to_numpy(), bins)
    B = B.flatten()
    D = B.shape
    Q = np.array([random_queries(D) for i in range(q_len)])

    A = better_mwem(B, D, Q, T, eps, part_avg=0.5)
    print(max(A), min(A), min(A) / max(A))
    errors = [np.abs(run(q, B) - run(q, A)) / 30161 for q in Q]
    print(max(errors), np.mean(errors))

    A /= sum(A)
    syn_train = sample_syn_data(A, 50000)

    headers = train_data.columns.to_list()
    bins.reverse()
    headers.reverse()
    X_syn = pd.DataFrame()

    for i in range(len(bins)):
        X_syn[headers[i]] = syn_train % (len(bins[i])-1) + 1
        syn_train = np.floor(syn_train / (len(bins[i])-1))
    X_syn = X_syn.iloc[:, ::-1]
    X_syn[X_syn.columns] = X_syn[X_syn.columns].applymap(np.int64)
    X_syn["income"] = X_syn["income"] - 1
    return X_syn

train_data = process_data("adult_data_clean.csv")
test_data = process_data("adult_test_clean.csv")

train_data = train_data[train_data.columns[0:10]]
test_data = test_data[test_data.columns[0:10]]


ann_cm = run_ann(train_data, test_data)
dt_cm = run_dt(train_data, test_data)
print((dt_cm[0][0] + dt_cm[1][1]) / np.sum(dt_cm))
knn_cm = run_knn(train_data, test_data)
print((knn_cm[0][0] + knn_cm[1][1]) / np.sum(knn_cm))

syn_data = run_mwem(pd.concat([train_data, test_data]), 30, 0.1, 5000)

dt_syn_cm = run_dt(syn_data, test_data)
print((dt_syn_cm[0][0] + dt_syn_cm[1][1]) / np.sum(dt_syn_cm))

knn_syn_cm = run_knn(syn_data, test_data)
print((knn_syn_cm[0][0] + knn_syn_cm[1][1]) / np.sum(knn_syn_cm))

ann_syn_cm = run_ann(syn_data, test_data, epochs=200)
print((ann_syn_cm[0][0] + ann_syn_cm[1][1]) / np.sum(ann_syn_cm))





dt_syn_cm = run_dt(pd.concat([train_data, syn_data[:100]]), test_data)
print((dt_syn_cm[0][0] + dt_syn_cm[1][1]) / np.sum(dt_syn_cm))

knn_syn_cm = run_knn(pd.concat([train_data, syn_data[:100]]), test_data)
print((knn_syn_cm[0][0] + knn_syn_cm[1][1]) / np.sum(knn_syn_cm))

ann_syn_cm = run_ann(syn_data, pd.concat([train_data, test_data]), epochs=200)
print((ann_syn_cm[0][0] + ann_syn_cm[1][1]) / np.sum(ann_syn_cm))


