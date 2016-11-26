import pickle
import pandas as pd
from xgboost import XGBClassifier


with open('./data/samples_dict_list.pickle', 'rb') as handle:
    samples = pickle.load(handle)

df = pd.DataFrame(samples)
df = df.set_index(['contour'])

y_train = df.is_plan

features = ['avc1', 'avc2', 'avc3', 'cx', 'cy', 'extent', 'm00', 'rect_extent', 'top_bin1', 'top_bin2', 'top_bin3', 'top_bin4', 'top_bin5', 'white_balance']

x_train = df[features]

model = XGBClassifier()
model.fit(x_train, y_train)


cor_mat = x_train.corr()
threshold = 0.7
important_corrs = (cor_mat[abs(cor_mat) > threshold][cor_mat != 1.0]).unstack().dropna().to_dict()
unique_important_corrs = pd.DataFrame(list(set([(tuple(sorted(key)), important_corrs[key]) for key in important_corrs])), columns=['attr_pairs','correlation'])



model = XGBClassifier()
