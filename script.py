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

train = pd.read_csv("adult_data_clean.csv")
train = train.drop("fnlwgt",1)
train = train.drop("capital-gain",1)
train = train.drop("capital-loss",1)
train = train.drop("age",1)
train = train.drop("hours-per-week",1)
train = train.drop("native-country",1)
train = train.drop("education",1)
train = train.drop("education-num",1)
X_train = train[train.columns[:-1]]
X_train = preprocessor.fit_transform(X_train).toarray().astype(int)
y_train = train[train.columns[-1]]
y_train = LabelEncoder().fit_transform(y_train)

test = pd.read_csv("adult_test_clean.csv")
test = test.drop("fnlwgt",1)
test = test.drop("capital-gain",1)
test = test.drop("capital-loss",1)
test = test.drop("age",1)
test = test.drop("hours-per-week",1)
test = test.drop("native-country",1)
test = test.drop("education",1)
test = test.drop("education-num",1)
X_test = test[test.columns[:-1]]
X_test = preprocessor.fit_transform(X_test).toarray().astype(int)
y_test = test[test.columns[-1]]
y_test = LabelEncoder().fit_transform(y_test)

classifier = Sequential()
classifier.add(Dense(24, activation = 'relu', input_dim = 41))
classifier.add(Dense(12, activation = 'relu'))
classifier.add(Dense(6, activation = 'relu'))
classifier.add(Dense(1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = 10, epochs = 50)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
ann_cm = confusion_matrix(y_test, y_pred)

dt_clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=5, min_samples_leaf=5)
dt_clf_gini.fit(X_train, y_train)
y_pred_gini = dt_clf_gini.predict(X_test)
dt_cm = confusion_matrix(y_test, y_pred_gini)

# MWEM

train = pd.read_csv("adult_data_clean.csv")
train = train.drop("fnlwgt",1)
train = train.drop("capital-gain",1)
train = train.drop("capital-loss",1)
train = train.drop("age",1)
train = train.drop("hours-per-week",1)
train = train.drop("native-country",1)
train = train.drop("education",1)
train = train.drop("education-num",1)

train.replace(["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"], [1,2,3,4,5,6,7,8], inplace=True, regex=True)
#train.replace(["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], inplace=True, regex=True)
train.replace(["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"], [1,2,3,4,5,6,7], inplace=True, regex=True)
train.replace(["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"], [1,2,3,4,5,6,7,8,9,10,11,12,13,14], inplace=True, regex=True)
train.replace(["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"], [1,2,3,4,5,6], inplace=True, regex=True)
train.replace(["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"], [1,2,3,4,5], inplace=True, regex=True)
train.replace(["Female", "Male"], [1,2], inplace=True, regex=True)
#train.replace(["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong"], [i for i in range(1,41)], inplace=True, regex=True)

X_train = train[train.columns[:-1]]
y_train = train[train.columns[-1]]
y_train = LabelEncoder().fit_transform(y_train)
bins = [[i for i in range(1,9)], [i for i in range(1,9)], [i for i in range(1,16)], [i for i in range(1,8)], [i for i in range(1,7)], [i for i in range(1,4)]]
B, _ = np.histogramdd(X_train.to_numpy(), bins)

B = B.flatten()
D = B.shape
Q = np.array([random_queries(D) for i in range(200)])
T = 100
eps = 0.2

A = better_mwem(B, D, Q, T, eps)
errors = [np.abs(run(q, B) - run(q, A)) / 30161 for q in Q]

# Now sample from A and run the ANN / Decision Tree again. Then compare the results.