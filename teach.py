import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data_directory = './data'

with open(data_directory + '/samples_dict_list.pickle', 'rb') as handle:
    samples = pickle.load(handle)

df = pd.DataFrame(samples)
df = df.set_index(['contour'])

features = ['avc1', 'avc2', 'avc3', 'cx', 'cy', 'extent', 'm00', 'rect_extent', 'top_bin1', 'top_bin2', 'top_bin3', 'top_bin4', 'top_bin5', 'white_balance']
x_train = df[features]
y_train = df.is_plan

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)

with open(data_directory + '/rf_wo_moments2.pickle', 'wb') as handle:
    pickle.dump(rf, handle)