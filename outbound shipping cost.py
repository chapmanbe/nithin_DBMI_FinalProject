import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import scipy
from sklearn.utils import check_array

np.set_printoptions(suppress=True) 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df_january = pd.read_pickle('january.pkl')
df_february = pd.read_pickle('february.pkl')
df_march = pd.read_pickle('march.pkl')
df_april = pd.read_pickle('april.pkl')
df_may = pd.read_pickle('may.pkl')
df_june = pd.read_pickle('june.pkl')
df_july = pd.read_pickle('july.pkl')
df_august = pd.read_pickle('august.pkl')
df_september = pd.read_pickle('september.pkl')
df_october = pd.read_pickle('october.pkl')
df_november = pd.read_pickle('november.pkl')
df_december = pd.read_pickle('december.pkl')

df = pd.concat([df_july, df_august, df_september, df_october, df_november, \
                df_december, df_january, df_february, df_march, df_april, df_may, df_june], \
               ignore_index=True)

df.count()
df.isnull().sum()
df.head()
df.tail()
df.dtypes
df = df.dropna()
df = df.drop_duplicates()
df = df.reset_index()
df = df.drop('index', axis = 1)
df['pkg_id'] = df['pkg_id'].astype(int).astype(str)
df['sls_trans_line_id'] = df['sls_trans_line_id'].astype(int).astype(str)
df['partner_num'] = df['partner_num'].astype(int).astype(str)
df['calc_zone_num'] = df['calc_zone_num'].astype(int)
df['item_qty'] = df['item_qty'].astype(int)
df.describe()
pd.set_option('display.float_format', lambda x: '%.2f' % x)
df.describe()

df = df[~(np.logical_or(np.logical_or(df['item_dim_length_meas'] == 0, \
                               df['item_dim_width_meas'] == 0), \
                 df['item_dim_height_meas'] == 0))]
df = df[~(df['full_sku_weight_meas'] == 0)]
df = df[~(df['pkg_chrg_amt'] <= 0)]
df = df.reset_index()
df = df.drop('index', axis = 1)
df.columns
plt.boxplot(df['item_dim_length_meas'])
plt.show()
df = df[~(df['item_dim_length_meas'] > 150)]
plt.boxplot(df['item_dim_width_meas'])
plt.show()
df = df[~(df['item_dim_width_meas'] > 150)]
plt.boxplot(df['item_dim_height_meas'])
plt.show()
df = df[~(df['item_dim_height_meas'] > 150)]
plt.boxplot(df['calc_wght_amt'])
plt.show()
plt.boxplot(df['full_sku_weight_meas'])
plt.show()
df = df[~(np.logical_or(df['full_sku_weight_meas'] > (np.mean(df['full_sku_weight_meas'])+(4*np.std(df['full_sku_weight_meas']))), \
                 df['calc_wght_amt'] > (np.mean(df['calc_wght_amt'])+(4*np.std(df['calc_wght_amt'])))))]
df = df[~(np.logical_or(df['full_sku_weight_meas'] <= 1, df['calc_wght_amt'] <= 1))] 
plt.boxplot(df['pkg_chrg_amt'])
plt.show()
df = df[~(np.logical_or(df['pkg_chrg_amt'] > (np.mean(df['pkg_chrg_amt'])+(4*np.std(df['pkg_chrg_amt']))), \
                df['pkg_chrg_amt'] < 1))]
df = df[~(np.logical_or(np.logical_or(df['item_dim_length_meas'] == 1, \
                  df['item_dim_width_meas'] == 1), \
                  df['item_dim_height_meas'] == 1))]
df = df.reset_index()
df = df.drop('index', axis = 1)
list(df.columns)
df.corr()
df[['item_dim_length_meas','item_dim_width_meas', 'item_dim_height_meas', 'pkg_chrg_amt']].head()
df_corr = df.corr()

%matplotlib inline
sns.heatmap(df_corr)
plt.show()

zone_dummies = pd.get_dummies(df.calc_zone_num, prefix = 'zone').iloc[:, 1:]
df = pd.concat([df, zone_dummies], axis = 1)
carr_name_dummies = pd.get_dummies(df.calc_carr_grp_name, prefix = 'carr_name').iloc[:, 1:]
df = pd.concat([df, carr_name_dummies], axis = 1)
orgn_dummies = pd.get_dummies(df.orgn_terr_cd, prefix = 'origin').iloc[:, 1:]
df = pd.concat([df, orgn_dummies], axis = 1)
dstn_dummies = pd.get_dummies(df.dstn_terr_cd, prefix = 'destination').iloc[:,1:]
df = pd.concat([df, dstn_dummies], axis = 1)
f_cols = ['item_dim_length_meas',
 'item_dim_width_meas',
 'item_dim_height_meas',
 'calc_wght_amt']
 X = df[f_cols]
 y = df['pkg_chrg_amt']
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 40)
train = df.sample(frac=0.8)
test = df.loc[~df.index.isin(train.index)]

print(train.shape)
print(test.shape)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
lm_all = LinearRegression()
lm_all.fit(train[f_cols], train.pkg_chrg_amt)
lm_all_pred = lm_all.predict(test[f_cols])
list(zip(f_cols, lm_all.coef_))
print(lm_all.score(train[f_cols], train['pkg_chrg_amt']))
print(mean_squared_error(lm_all_pred, test['pkg_chrg_amt']))
print(sqrt(mean_squared_error(lm_all_pred, test['pkg_chrg_amt'])))

rf_all = RandomForestRegressor()
rf_all.fit(train[f_cols], train.pkg_chrg_amt)
rf_all_pred = rf_all.predict(test[f_cols])

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(rf.score(X_test, y_test))
print(np.sqrt(mean_squared_error(y_test, y_pred)))
print(rf_all.score(train[f_cols], train['pkg_chrg_amt']))
print(rf_all.score(test[f_cols], test['pkg_chrg_amt']))
print(mean_squared_error(rf_all_pred, test['pkg_chrg_amt']))
print(sqrt(mean_squared_error(rf_all_pred, test['pkg_chrg_amt'])))


gb_all = GradientBoostingRegressor()
gb_all.fit(train[f_cols], train.pkg_chrg_amt)
gb_all_pred = gb_all.predict(test[f_cols])

print(gb_all.score(train[f_cols], train['pkg_chrg_amt']))
print(gb_all.score(test[f_cols], test['pkg_chrg_amt']))
print(mean_squared_error(gb_all_pred, test['pkg_chrg_amt']))
print(sqrt(mean_squared_error(gb_all_pred, test['pkg_chrg_amt'])))


rf_all.get_params
gb_all.get_params
rf_all = RandomForestRegressor(n_estimators=20)
rf_all.fit(train[f_cols], train.pkg_chrg_amt)
rf_all_pred = rf_all.predict(test[f_cols])


print(rf_all.score(train[f_cols], train['pkg_chrg_amt']))
print(rf_all.score(test[f_cols], test['pkg_chrg_amt']))
print(mean_squared_error(rf_all_pred, test['pkg_chrg_amt']))
print(sqrt(mean_squared_error(rf_all_pred, test['pkg_chrg_amt'])))

rf_all = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score = True)
rf_all.fit(train[f_cols], train.pkg_chrg_amt)
rf_all_pred = rf_all.predict(test[f_cols])

print(rf_all.score(train[f_cols], train['pkg_chrg_amt']))
print(rf_all.score(test[f_cols], test['pkg_chrg_amt']))
print(mean_squared_error(rf_all_pred, test['pkg_chrg_amt']))
print(sqrt(mean_squared_error(rf_all_pred, test['pkg_chrg_amt'])))
print(mean_absolute_error(rf_all_pred, test['pkg_chrg_amt']))

all_predictions = rf_all.predict(df[f_cols])
list(zip(df['pkg_chrg_amt'], all_predictions))
test.shape
test['calc_wght_amt'].describe()


def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(((y_true - y_pred) / y_true)) * 100

y_true = test_10To15.pkg_chrg_amt
y_pred = rf_all_p_10To15
round(mean_absolute_percentage_error(y_true, y_pred),2)





