#%%
import numpy as np
import pandas as pd
import snsynth
#%%
pums = pd.read_csv("2018a.csv", index_col=0) # in datasets/
nf = (np.nan_to_num(pums.to_numpy()).astype(int)//5)*5 # round to 5

synth = snsynth.MWEMSynthesizer(epsilon=1.0, splits=[[0,1], [2,3,2], [4,5,6], [7,8], [9,10], [11]]) 
synth.fit(nf)

sample = synth.sample(10) # get 10 synthetic rows
print(sample)
# %%
