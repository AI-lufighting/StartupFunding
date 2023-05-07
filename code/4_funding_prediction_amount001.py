#!/usr/bin/env python
# coding: utf-8

# # 6 融资金额关键因素分析与预测
import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

def result_metrics(y_true,y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print('RMSE,MAE,MAPE: {} {} {}'.format(rmse,mae,mape))

    return rmse,mae,mape

### 只留下有融资金额的公司进行金额预测
data_se_all_reg = pd.read_csv('data_se_all_reg001.csv').drop('profile',axis=1)

data_X_f, data_Y_fl = data_se_all_reg.drop('funding',axis=1),data_se_all_reg['funding']


from matplotlib import pyplot as plt
plt.hist(data_Y_fl)
plt.title('The distribution of log Y')
plt.show()


# ## 6.1 关键因素分析

## ALL Market

from statsmodels.stats.outliers_influence import variance_inflation_factor
def vif_C(data_XX):
    def vif_count(xnames,data_vv):
        x= xnames
        v_array = np.array(data_vv)
        n = len(x)
        vifs = []
        for i in range(n):
            vif = variance_inflation_factor(v_array,i)
            # print(x[i],vif)
            vifs.append((vif,x[i]))

        vifs = sorted(vifs)
        return vifs[-1]
    data_vt = data_XX
    xs = list(data_XX.columns)
    k = 10
    t = 1000
    print(k)
    drop_name = []
    while t>100:
        vix = vif_count(xs,data_vt)
        vifx,cname = vix
        if vifx>k:
            t-=1
            xs.remove(cname)
            data_vt = data_vt.drop(labels = [cname], axis = 1)
            print(cname,vifx)
            drop_name.append(cname)
        else:
            break
    print(drop_name)
    return drop_name



drop_name = vif_C(data_X_f)

data_X_l = data_X_f.drop(labels=drop_name,axis=1)


## forward
def plot_BIC(models_best):
    num_v = len(models_best["model"])
    print('plot BIC of different size model')

    Y=[]
    fig = plt.figure(figsize=(6,2))
    for i in range(1,num_v):
        Y.append(models_best["model"][i].bic)
    print('The number of optimal variables determined using the BIC:',(np.argmin(Y)+1))
    plt.plot(range(1,num_v),Y,c='black')
    plt.scatter(np.argmin(Y)+1,Y[np.argmin(Y)],c='r')
    plt.show()
def processSubset(data_X,data_y,combo):
    X_train1 = sm.add_constant(data_X[combo])
    model = sm.OLS(data_y,X_train1)
    model = model.fit()
    RSS= ((model.predict(X_train1) - data_y) ** 2).sum()
    return {"model":model, "RSS":RSS, 'Index':combo}
def forward(predictors,data_X,data_y):
    # Pull out predictors we still need to process
    remaining_predictors = [p for p in data_X.columns if p not in predictors]
    results = []
    for p in remaining_predictors:
        results.append(processSubset(data_X,data_y,predictors+[p]))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    # Return the best model, along with some other useful information about the model
    return best_model


regfit_fwd_b = pd.DataFrame(columns=["RSS", "model","Index"])
predictors = []
for i in range(1,len(data_X_l.columns)+1):   
    regfit_fwd_b.loc[i] = forward(predictors,data_X_l,data_Y_fl)
    predictors = regfit_fwd_b.loc[i]["Index"]
plot_BIC(regfit_fwd_b)

num_v = 17
print(regfit_fwd_b['model'][num_v].summary())


markets = ['markets_SaaS', 'markets_Technology', 'markets_Mobile',
       'markets_Enterprise_Software', 'markets_E-Commerce',
       'markets_Mobile_Application', 'markets_Healthcare', 'markets_Education',
       'markets_Web_Development', 'markets_Artificial_Intelligence',
       'markets_Software', 'markets_Digital_Marketing', 'markets_B2B',
       'markets_Social_Media', 'markets_Sales_and_Marketing',
       'markets_Machine_Learning', 'markets_Health_and_Wellness',
       'markets_Blockchain_/_Cryptocurrency', 'markets_Fin_Tech',
       'markets_Marketplaces']
markets_df = data_X_f[markets]
data_X_without_market = data_X_f.drop(markets,axis=1)


markets_df.sum().sort_values().plot.barh(color=plt.cm.Paired(np.arange(len(markets_df))))



# ## markets_SaaS

data_X_SaaS = data_X_without_market[markets_df['markets_SaaS']>0]
data_Y_SaaS = data_Y_fl[markets_df['markets_SaaS']>0]


regfit_fwd_SaaS = pd.DataFrame(columns=["RSS", "model","Index"])
predictors = []
for i in range(1,len(data_X_SaaS.columns)+1):   
    regfit_fwd_SaaS.loc[i] = forward(predictors,data_X_SaaS,data_Y_SaaS)
    predictors = regfit_fwd_SaaS.loc[i]["Index"]
plot_BIC(regfit_fwd_SaaS)


num_v = 6
print(regfit_fwd_SaaS['model'][num_v].summary())


# ## markets_Enterprise_Software 


data_X_Enterprise_Software = data_X_without_market[markets_df['markets_Enterprise_Software']>0]
data_Y_Enterprise_Software = data_Y_fl[markets_df['markets_Enterprise_Software']>0]



regfit_fwd_Enterprise_Software = pd.DataFrame(columns=["RSS", "model","Index"])
predictors = []
for i in range(1,len(data_X_Enterprise_Software.columns)+1):   
    regfit_fwd_Enterprise_Software.loc[i] = forward(predictors,data_X_Enterprise_Software,data_Y_Enterprise_Software)
    predictors = regfit_fwd_Enterprise_Software.loc[i]["Index"]
plot_BIC(regfit_fwd_Enterprise_Software)



num_v = 3
print(regfit_fwd_Enterprise_Software['model'][num_v].summary())



# ## 融资金额预测



train_data_reg = pd.read_csv('train_data001_reg.csv').drop('profile',axis=1)
test_data_reg = pd.read_csv('test_data001_reg.csv').drop('profile',axis=1)
X_train, X_test, y_train, y_test = train_data_reg.drop('funding',axis=1),test_data_reg.drop('funding',axis=1),train_data_reg['funding'],test_data_reg['funding']
X_train.shape,X_test.shape




X_train_d_e = X_train.iloc[:,7:]
X_test_d_e = X_test.iloc[:,7:]
X_train_d_es = X_train.iloc[:,9:]
X_test_d_es = X_test.iloc[:,9:]


# ## 5.1 决策树

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
estimator = DecisionTreeRegressor(random_state =0)
param_grid = [{'max_depth':[2,5,10,20]}]
grid_search = GridSearchCV(estimator, param_grid, cv=5,scoring='neg_mean_squared_error')

grid_search.fit(X_train,y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(grid_search.best_params_)
model_dt_best = grid_search.best_estimator_

print('#################################')
print('Result on training set:')
y_test_pred = model_dt_best.predict(X=X_train) 
result_dt = result_metrics(y_train, y_test_pred)
print('#################################')
print('Result on test set:')
y_test_pred = model_dt_best.predict(X=X_test) 
result_dt = result_metrics(y_test, y_test_pred)
print('#################################')

## 不加情绪和情感
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
estimator = DecisionTreeRegressor(random_state =0)
param_grid = [{'max_depth':[2,5,10,20]}]
grid_search = GridSearchCV(estimator, param_grid, cv=5,scoring='neg_mean_squared_error')

grid_search.fit(X_train_d_es,y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(grid_search.best_params_)
model_dt_best = grid_search.best_estimator_

print('#################################')
print('Result on training set:')
y_test_pred = model_dt_best.predict(X=X_train_d_es) 
result_dt = result_metrics(y_train, y_test_pred)
print('#################################')
print('Result on test set:')
y_test_pred = model_dt_best.predict(X=X_test_d_es) 
result_dt = result_metrics(y_test, y_test_pred)
print('#################################')

## 不加情绪
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
estimator = DecisionTreeRegressor(random_state =0)
param_grid = [{'max_depth':[2,5,10,20]}]
grid_search = GridSearchCV(estimator, param_grid, cv=5,scoring='neg_mean_squared_error')

grid_search.fit(X_train_d_e,y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(grid_search.best_params_)
model_dt_best = grid_search.best_estimator_

print('#################################')
print('Result on training set:')
y_test_pred = model_dt_best.predict(X=X_train_d_e) 
result_dt = result_metrics(y_train, y_test_pred)
print('#################################')
print('Result on test set:')
y_test_pred = model_dt_best.predict(X=X_test_d_e) 
result_dt = result_metrics(y_test, y_test_pred)
print('#################################')

# ## 5.2 随机森林

from sklearn.ensemble import RandomForestRegressor
param_grid = [{'n_estimators':[10,100],'max_depth':[2,5,10]}]
model_rf = RandomForestRegressor(random_state =0)
grid_search = GridSearchCV(model_rf, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(X_train,y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(grid_search.best_params_)
model_rf_best = grid_search.best_estimator_
print('#################################')
print('Result on training set:')
y_train_pred = model_rf_best.predict(X=X_train) 
result_rf = result_metrics(y_train, y_train_pred)
print('#################################')
print('Result on test set:')
y_test_pred = model_rf_best.predict(X=X_test) 
result_rf = result_metrics(y_test, y_test_pred)
print('#################################')

## 不加情绪和情感
from sklearn.ensemble import RandomForestRegressor
param_grid = [{'n_estimators':[10,100],'max_depth':[2,5,10]}]
model_rf = RandomForestRegressor(random_state =0)
grid_search = GridSearchCV(model_rf, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(X_train_d_es,y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(grid_search.best_params_)
model_rf_best = grid_search.best_estimator_
print('#################################')
print('Result on training set:')
y_train_pred = model_rf_best.predict(X=X_train_d_es) 
result_rf = result_metrics(y_train, y_train_pred)
print('#################################')
print('Result on test set:')
y_test_pred = model_rf_best.predict(X=X_test_d_es) 
result_rf = result_metrics(y_test, y_test_pred)
print('#################################')

## 不加情绪
from sklearn.ensemble import RandomForestRegressor
param_grid = [{'n_estimators':[10,100],'max_depth':[2,5,10]}]
model_rf = RandomForestRegressor(random_state =0)
grid_search = GridSearchCV(model_rf, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(X_train_d_e,y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(grid_search.best_params_)
model_rf_best = grid_search.best_estimator_
print('#################################')
print('Result on training set:')
y_train_pred = model_rf_best.predict(X=X_train_d_e) 
result_rf = result_metrics(y_train, y_train_pred)
print('#################################')
print('Result on test set:')
y_test_pred = model_rf_best.predict(X=X_test_d_e) 
result_rf = result_metrics(y_test, y_test_pred)
print('#################################')


# ## 5.3 AdaBoost

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

param_grid = {
              "n_estimators": [10, 50,100,200]
             }
dt = DecisionTreeRegressor(max_depth=1,random_state=0)
model_ab = AdaBoostRegressor(base_estimator=dt, random_state=0)
grid_search = GridSearchCV(model_ab, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(X_train,y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(grid_search.best_params_)
model_ab_best = grid_search.best_estimator_


print('#################################')
print('Result on training set:')
y_train_pred = model_ab_best.predict(X=X_train) 
result_logistic = result_metrics(y_train, y_train_pred)
print('#################################')
print('Result on test set:')
y_test_pred = model_ab_best.predict(X=X_test) 
result_logistic = result_metrics(y_test, y_test_pred)
print('#################################')


#不添加情绪和情感
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

param_grid = {
              "n_estimators": [10, 50,100,200]
             }
dt = DecisionTreeRegressor(max_depth=1,random_state=0)
model_ab = AdaBoostRegressor(base_estimator=dt, random_state=0)
grid_search = GridSearchCV(model_ab, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(X_train_d_es,y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(grid_search.best_params_)
model_ab_best = grid_search.best_estimator_


print('#################################')
print('Result on training set:')
y_train_pred = model_ab_best.predict(X=X_train_d_es) 
result_logistic = result_metrics(y_train, y_train_pred)
print('#################################')
print('Result on test set:')
y_test_pred = model_ab_best.predict(X=X_test_d_es) 
result_logistic = result_metrics(y_test, y_test_pred)
print('#################################')


#不添加情绪
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

param_grid = {
              "n_estimators": [10, 50,100,200]
             }
dt = DecisionTreeRegressor(max_depth=1,random_state=0)
model_ab = AdaBoostRegressor(base_estimator=dt, random_state=0)
grid_search = GridSearchCV(model_ab, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(X_train_d_e,y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(grid_search.best_params_)
model_ab_best = grid_search.best_estimator_


print('#################################')
print('Result on training set:')
y_train_pred = model_ab_best.predict(X=X_train_d_e) 
result_logistic = result_metrics(y_train, y_train_pred)
print('#################################')
print('Result on test set:')
y_test_pred = model_ab_best.predict(X=X_test_d_e) 
result_logistic = result_metrics(y_test, y_test_pred)
print('#################################')


# ## 5.4 GBDT

from sklearn.ensemble import GradientBoostingRegressor
param_grid = {
              "n_estimators": [10, 50,100,200]
             }
model_gbdt = GradientBoostingRegressor(learning_rate=0.1, random_state=0)
grid_search = GridSearchCV(model_gbdt, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(X_train,y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(grid_search.best_params_)
model_gbdt_best = grid_search.best_estimator_




print('#################################')
print('Result on training set:')
y_train_pred = model_gbdt_best.predict(X=X_train) 
result_logistic = result_metrics(y_train, y_train_pred)
print('#################################')
print('Result on test set:')
y_test_pred = model_gbdt_best.predict(X=X_test) 
result_logistic = result_metrics(y_test, y_test_pred)
print('#################################')

#不添加情绪和情感
from sklearn.ensemble import GradientBoostingRegressor
param_grid = {
              "n_estimators": [10, 50,100,200]
             }
model_gbdt = GradientBoostingRegressor(learning_rate=0.1, random_state=0)
grid_search = GridSearchCV(model_gbdt, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(X_train_d_es,y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(grid_search.best_params_)
model_gbdt_best = grid_search.best_estimator_




print('#################################')
print('Result on training set:')
y_train_pred = model_gbdt_best.predict(X=X_train_d_es) 
result_logistic = result_metrics(y_train, y_train_pred)
print('#################################')
print('Result on test set:')
y_test_pred = model_gbdt_best.predict(X=X_test_d_es) 
result_logistic = result_metrics(y_test, y_test_pred)
print('#################################')

#不添加情绪
from sklearn.ensemble import GradientBoostingRegressor
param_grid = {
              "n_estimators": [10, 50,100,200]
             }
model_gbdt = GradientBoostingRegressor(learning_rate=0.1, random_state=0)
grid_search = GridSearchCV(model_gbdt, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(X_train_d_e,y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(grid_search.best_params_)
model_gbdt_best = grid_search.best_estimator_

print('#################################')
print('Result on training set:')
y_train_pred = model_gbdt_best.predict(X=X_train_d_e) 
result_logistic = result_metrics(y_train, y_train_pred)
print('#################################')
print('Result on test set:')
y_test_pred = model_gbdt_best.predict(X=X_test_d_e) 
result_logistic = result_metrics(y_test, y_test_pred)
print('#################################')

# ## 5.5 SVM
from sklearn import svm
param_grid = {'kernel' :['rbf','linear']}
model_svm = svm.SVR()
grid_search = GridSearchCV(model_svm, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(X_train,y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(grid_search.best_params_)
model_svm_best = grid_search.best_estimator_

print('#################################')
print('Result on training set:')
y_train_pred = model_svm_best.predict(X=X_train) 
result_logistic = result_metrics(y_train, y_train_pred)
print('#################################')
print('Result on test set:')
y_test_pred = model_svm_best.predict(X=X_test) 
result_logistic = result_metrics(y_test, y_test_pred)
print('#################################')

#不添加情绪和情感
from sklearn import svm
param_grid = {'kernel' :['rbf','linear']}
model_svm = svm.SVR()
grid_search = GridSearchCV(model_svm, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(X_train_d_es,y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(grid_search.best_params_)
model_svm_best = grid_search.best_estimator_

print('#################################')
print('Result on training set:')
y_train_pred = model_svm_best.predict(X=X_train_d_es) 
result_logistic = result_metrics(y_train, y_train_pred)
print('#################################')
print('Result on test set:')
y_test_pred = model_svm_best.predict(X=X_test_d_es) 
result_logistic = result_metrics(y_test, y_test_pred)
print('#################################')

#不添加情绪
from sklearn import svm
param_grid = {'kernel' :['rbf','linear']}
model_svm = svm.SVR()
grid_search = GridSearchCV(model_svm, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(X_train_d_e,y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(grid_search.best_params_)
model_svm_best = grid_search.best_estimator_

print('#################################')
print('Result on training set:')
y_train_pred = model_svm_best.predict(X=X_train_d_e) 
result_logistic = result_metrics(y_train, y_train_pred)
print('#################################')
print('Result on test set:')
y_test_pred = model_svm_best.predict(X=X_test_d_e) 
result_logistic = result_metrics(y_test, y_test_pred)
print('#################################')

# ## 5.6 Deep learning (MLP)

from sklearn.neural_network import MLPRegressor
from warnings import filterwarnings
filterwarnings('ignore')
param_grid = {'max_iter':[50],'batch_size':[10,50]}
model_mlp = MLPRegressor(hidden_layer_sizes=(30,10), activation='relu',learning_rate_init=0.01, solver='adam', random_state=0)
grid_search = GridSearchCV(model_mlp, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(X_train,y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(grid_search.best_params_)
model_mlp_best = grid_search.best_estimator_


print('#################################')
print('Result on training set:')
y_train_pred = model_mlp_best.predict(X=X_train) 
result_logistic = result_metrics(y_train, y_train_pred)
print('#################################')
print('Result on test set:')
y_test_pred = model_mlp_best.predict(X=X_test) 
result_logistic = result_metrics(y_test, y_test_pred)
print('#################################')


# 不添加情绪和情感
from sklearn.neural_network import MLPRegressor
from warnings import filterwarnings
filterwarnings('ignore')
param_grid = {'max_iter':[50],'batch_size':[10,50]}
model_mlp = MLPRegressor(hidden_layer_sizes=(30,10), activation='relu',learning_rate_init=0.01, solver='adam', random_state=0)
grid_search = GridSearchCV(model_mlp, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(X_train_d_es,y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(grid_search.best_params_)
model_mlp_best = grid_search.best_estimator_


print('#################################')
print('Result on training set:')
y_train_pred = model_mlp_best.predict(X=X_train_d_es) 
result_logistic = result_metrics(y_train, y_train_pred)
print('#################################')
print('Result on test set:')
y_test_pred = model_mlp_best.predict(X=X_test_d_es) 
result_logistic = result_metrics(y_test, y_test_pred)
print('#################################')


# 不添加情绪
from sklearn.neural_network import MLPRegressor
from warnings import filterwarnings
filterwarnings('ignore')
param_grid = {'max_iter':[50],'batch_size':[10,50]}
model_mlp = MLPRegressor(hidden_layer_sizes=(30,10), activation='relu',learning_rate_init=0.01, solver='adam', random_state=0)
grid_search = GridSearchCV(model_mlp, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(X_train_d_e,y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(grid_search.best_params_)
model_mlp_best = grid_search.best_estimator_


print('#################################')
print('Result on training set:')
y_train_pred = model_mlp_best.predict(X=X_train_d_e) 
result_logistic = result_metrics(y_train, y_train_pred)
print('#################################')
print('Result on test set:')
y_test_pred = model_mlp_best.predict(X=X_test_d_e) 
result_logistic = result_metrics(y_test, y_test_pred)
print('#################################')
