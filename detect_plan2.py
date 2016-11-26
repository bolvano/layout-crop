In [2]: %pylab
Using matplotlib backend: MacOSX
Populating the interactive namespace from numpy and matplotlib

In [3]: important_features = ['mu02','nu20','m10','m12','m00', 'mu11', 'mu03']

In [4]: x = df[important_features]

In [5]: pd.tools.plotting.scatter_matrix(x)

In [7]: cor_mat = x.corr()

In [8]: threshold = 0.7

In [9]: important_corrs = (cor_mat[abs(cor_mat) > threshold][cor_mat != 1.0]).unstack().dropna().to_dict()

In [10]: important_corrs
Out[10]: 
{('m00', 'm10'): 0.9703088466725297,
 ('m00', 'm12'): 0.91546758109814941,
 ('m00', 'mu02'): 0.89770516100611852,
 ('m10', 'm00'): 0.9703088466725297,
 ('m10', 'm12'): 0.96597169392222182,
 ('m10', 'mu02'): 0.93339815742681553,
 ('m12', 'm00'): 0.91546758109814941,
 ('m12', 'm10'): 0.96597169392222182,
 ('m12', 'mu02'): 0.94552643993077501,
 ('mu02', 'm00'): 0.89770516100611852,
 ('mu02', 'm10'): 0.93339815742681553,
 ('mu02', 'm12'): 0.94552643993077501}

In [11]: unique_important_corrs = pd.DataFrame(
   ....: list(set([(tuple(sorted(key)), important_corrs[key]) \
   ....: for key in important_corrs])), columns=['attr_pairs','correlation'])

In [12]: cor_mat
Out[12]: 
          mu02      nu20       m10       m12       m00      mu11      mu03
mu02  1.000000 -0.173033  0.933398  0.945526  0.897705 -0.291439 -0.587732
nu20 -0.173033  1.000000 -0.051669  0.048879 -0.030144  0.072781  0.084060
m10   0.933398 -0.051669  1.000000  0.965972  0.970309 -0.113345 -0.330935
m12   0.945526  0.048879  0.965972  1.000000  0.915468 -0.180561 -0.446780
m00   0.897705 -0.030144  0.970309  0.915468  1.000000 -0.081034 -0.269963
mu11 -0.291439  0.072781 -0.113345 -0.180561 -0.081034  1.000000  0.587478
mu03 -0.587732  0.084060 -0.330935 -0.446780 -0.269963  0.587478  1.000000

In [41]: important_features
Out[41]: ['mu02', 'nu20', 'm10', 'm12', 'm00', 'mu11', 'mu03']

In [42]: all_important_features = ['mu02', 'nu20', 'm10', 'm12', 'm00', 'mu11', 'mu03', 'nu11', 'm01', 'mu12', 'nu02', 'm03', 'm11']

In [43]: 

In [43]: 

In [43]: regr4 = linear_model.LinearRegression()

In [44]: x4 = df[all_important_features]

In [45]: regr4.fit(x4, y)
Out[45]: LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

In [46]: with open('regr4.pickle', 'wb') as handle:
    pickle.dump(regr4, handle)
   ....:     

In [47]: df['predicted4'] = regr4.predict(df[all_important_features])

In [48]: 

In [48]: df[['is_plan','predicted3','predicted4']]