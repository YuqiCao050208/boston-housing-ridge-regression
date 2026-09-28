import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


data = fetch_openml(name='boston', version=1, as_frame=True)
data_pd = pd.DataFrame(data.data,columns=data.feature_names)
data_pd['price'] = data.target

data_pd.dtypes.value_counts()

data_pd['CHAS'] = data_pd['CHAS'].astype('float64')


data_pd['LSTAT_RM_product'] = data_pd['LSTAT'] * data_pd['RM']


data_pd['LSTAT_squared'] = data_pd['LSTAT'] ** 2


data_pd['CHAS_RM_product'] = data_pd['CHAS'] * data_pd['RM']


data_pd['CHAS_PTRATIO_product'] = data_pd['CHAS'] * data_pd['PTRATIO']


data_pd['CHAS_LSTAT_product'] = data_pd['CHAS'] * data_pd['LSTAT']

data_pd.head()


data_pd.corr()['price']


corr = data_pd.corr()
corr = corr['price']
corr[abs(corr)>0.5].sort_values().plot.bar()

data_pd = data_pd[['LSTAT','LSTAT_RM_product','LSTAT_squared','PTRATIO','RM','price']]
y = np.array(data_pd['price'])
data_pd=data_pd.drop(['price'],axis=1)
X = np.array(data_pd)

train_X,test_X,train_Y,test_Y = train_test_split(X,y,test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_X)
X_test_scaled = scaler.transform(test_X)

param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
grid_search.fit(X_train_scaled, train_Y)
print(f"the optimal alpha：{grid_search.best_params_['alpha']}")

ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train_scaled, train_Y)

y_predict = ridge_model.predict(X_test_scaled)

mse=metrics.mean_squared_error(y_predict,test_Y)
print(mse)

print(f"model coefficients：{ridge_model.coef_}")
