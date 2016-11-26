import pandas as pd
import numpy as np
import pickle
with open('./Downloads/samples.pickle', 'rb') as handle:
    samples = pickle.load(handle)
df = pd.DataFrame(samples)
df['filename_color'] = df['filename'] + '_' + df['color']
df = df.set_index(['filename_color'])
df = df.drop('filename', 1)
df = df.drop('color', 1)
df
import xgboost as xg
from xgboost import XGBClassifier
from matplotlib import pyplot
import xgboost as xgb
import operator
from matplotlib import pylab as plt
%paste
list(df.columns[1:])
features = list(df.columns[1:])
df.is_plan
y_train = df.is_plan
df.select_dtypes(include=['object']).columns
df.select_dtypes(include=['number']).columns
x_train = df[features]
ceate_feature_map(features)
xgb_params = {"objective": "reg:linear", "eta": 0.01, "max_depth": 8, "seed": 42, "silent": 1}
num_rounds = 1000
dtrain = xgb.DMatrix(x_train, label=y_train)
del(x_train)
for feat in df.select_dtypes(include=['object']).columns:
    
    
    pass
for feat in df.select_dtypes(include=['float']).columns:
    print(feat)
for feat in df.select_dtypes(include=['float']).columns:
    m = df.groupby([feat])['is_plan'].mean()
    print(feat,m)
for feat in df.select_dtypes(include=['float']).columns:
    m = df.groupby([feat])['is_plan'].mean()
    print(m)
df
model = XGBClassifier()
x_train = df[features]
model.fit(x_train, y_train)
print(model.feature_importances_)
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()
from xgboost import plot_importance
plot_importance(model)
pyplot.show()
from sklearn import datasets, linear_model
important_features = ['mu02','nu20','m10','m12','m00', 'mu11', 'mu03']
x = df[important_features]
y = df.is_plan
regr = linear_model.LinearRegression()
regr.fit(x, y)
with open('regr.pickle', 'wb') as handle:
    pickle.dump(regr, handle)

df['predicted'] = regr.predict(df[important_features])
df[['is_plan','predicted']]
%hist
