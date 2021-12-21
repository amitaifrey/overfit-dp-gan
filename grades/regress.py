#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
#%% Train without synthetic data
csv2018a = np.nan_to_num(pd.read_csv("2018a.csv", index_col=0).to_numpy()).astype(int) # in datasets/
csv2018b = np.nan_to_num(pd.read_csv("2018b.csv", index_col=0).to_numpy()).astype(int)
csv2019b = np.nan_to_num(pd.read_csv("2019b.csv", index_col=0).to_numpy()).astype(int)
csv2018a //= 5
csv2018a *= 5
csv2019b //= 5
csv2019b *= 5


X_train = np.concatenate((csv2018a[...,:-1], csv2018b[...,:-1]))
Y_train = np.concatenate((csv2018a[...,-1], csv2018b[...,-1]))
reg = LinearRegression().fit(X_train, Y_train)
print(reg.score(X_train, Y_train))
print(reg.coef_)
print(reg.intercept_)

#%%
X_test = csv2019b[...,:-1]
Y_test = csv2019b[...,-1]
print(mean_squared_error(Y_test, reg.predict(X_test)))
print(r2_score(Y_test, reg.predict(X_test)))
# %%


#%% Train with synthetic data
csv2018a = np.nan_to_num(pd.read_csv("2018a.csv", index_col=0).to_numpy()).astype(int) # in datasets/
csv2018b = np.nan_to_num(pd.read_csv("2018b.csv", index_col=0).to_numpy()).astype(int)
csv2019b = np.nan_to_num(pd.read_csv("2019b.csv", index_col=0).to_numpy()).astype(int)
csv2018a //= 5
csv2018a *= 5
csv2019b //= 5
csv2019b *= 5

with open("2018b.synth.pkl", "rb") as f:
    synth = pickle.load(f)
csv_synth = synth.sample(100)


X_train = np.concatenate((csv2018a[...,:-1], csv2018b[...,:-1], csv_synth[...,:-1]))
Y_train = np.concatenate((csv2018a[...,-1], csv2018b[...,-1], csv_synth[...,-1]))
reg = LinearRegression().fit(X_train, Y_train)
print(reg.score(X_train, Y_train))
print(reg.coef_)
print(reg.intercept_)

#%%
X_test = csv2019b[...,:-1]
Y_test = csv2019b[...,-1]
print(mean_squared_error(Y_test, reg.predict(X_test)))
print(r2_score(Y_test, reg.predict(X_test)))
# %%
