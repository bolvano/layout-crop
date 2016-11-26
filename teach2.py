import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data_directory = './data'

with open(data_directory + '/samples_dict_list_ext.pickle', 'rb') as handle:
    samples = pickle.load(handle)

df = pd.DataFrame(samples)
df = df.set_index(['contour'])

features = ['hist_correl', 'm00', 'white_balance']
x_train = df[features]
y_train = df.is_plan

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)

with open(data_directory + '/rf_min.pickle', 'wb') as handle:
    pickle.dump(rf, handle)