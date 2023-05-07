#!/usr/bin/env python
# coding: utf-8

"""
@author: Xiaolu
@time: 2023/5/1 13:51
"""
# # 5 融资是否成功预测

import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import mean_squared_error, make_scorer,confusion_matrix, accuracy_score, recall_score, precision_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.model_selection import train_test_split,cross_val_score


def result_metrics(y_true,y_pred):
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true,y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    print('Accuracy,Recall,Precision,F1 score,AUC:', accuracy,recall,precision,f1,auc)

    return f1,auc,accuracy,recall,precision

train_data = pd.read_csv('./train_data001.csv').drop('profile',axis=1)
test_data = pd.read_csv('./test_data001.csv').drop('profile',axis=1)


X_train, X_test, y_train, y_test = train_data.drop('funding',axis=1),test_data.drop('funding',axis=1),train_data['funding'],test_data['funding']

X_train_d_e = X_train.iloc[:,7:]
X_test_d_e = X_test.iloc[:,7:]
X_train_d_es = X_train.iloc[:,9:]
X_test_d_es = X_test.iloc[:,9:]


print(y_train.value_counts())
print(y_test.value_counts())


# ## 5.1 决策树


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
estimator = DecisionTreeClassifier(random_state =0)
param_grid = [{'max_depth':[2,5,10,20]}]
grid_search = GridSearchCV(estimator, param_grid, cv=5,scoring='roc_auc')

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




## 不添加情绪和情感
estimator = DecisionTreeClassifier(random_state =0)
grid_search = GridSearchCV(estimator, param_grid, cv=5,scoring='roc_auc')
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




## 不添加情绪
estimator = DecisionTreeClassifier(random_state =0)
grid_search = GridSearchCV(estimator, param_grid, cv=5,scoring='roc_auc')
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


from sklearn.ensemble import RandomForestClassifier
param_grid = [{'n_estimators':[10,100],'max_depth':[2,5,10]}]
model_rf = RandomForestClassifier(random_state =0)
grid_search = GridSearchCV(model_rf, param_grid, cv=5,scoring='roc_auc')
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




### 不添加情绪和情感
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(random_state =0)
grid_search = GridSearchCV(model_rf, param_grid, cv=5,scoring='roc_auc')
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


### 不添加情绪
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(random_state =0)
grid_search = GridSearchCV(model_rf, param_grid, cv=5,scoring='roc_auc')
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


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
param_grid = {
              "n_estimators": [10, 50,100,200]
             }
dt = DecisionTreeClassifier(max_depth=1,random_state=0)
model_ab = AdaBoostClassifier(base_estimator=dt, random_state=0)
grid_search = GridSearchCV(model_ab, param_grid, cv=5,scoring='roc_auc')
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




### 不添加情绪和情感
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
param_grid = {
              "n_estimators": [10, 50,100,200]
             }
dt = DecisionTreeClassifier(max_depth=1,random_state =0)
model_ab = AdaBoostClassifier(base_estimator=dt, random_state=0)
grid_search = GridSearchCV(model_ab, param_grid, cv=5,scoring='roc_auc')
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




### 不添加情绪
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
param_grid = {
              "n_estimators": [10, 50,100,200]
             }
dt = DecisionTreeClassifier(max_depth=1,random_state =0)
model_ab = AdaBoostClassifier(base_estimator=dt, random_state=0)
grid_search = GridSearchCV(model_ab, param_grid, cv=5,scoring='roc_auc')
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

from sklearn.ensemble import GradientBoostingClassifier
param_grid = {
              "n_estimators": [10, 50,100,200]
             }
model_gbdt = GradientBoostingClassifier(learning_rate=0.1, random_state=0)
grid_search = GridSearchCV(model_gbdt, param_grid, cv=5,scoring='roc_auc')
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

### 不添加情绪和情感
from sklearn.ensemble import GradientBoostingClassifier
param_grid = {
              "n_estimators": [10, 50,100,200]
             }
model_gbdt = GradientBoostingClassifier(learning_rate=0.1, random_state=0)
grid_search = GridSearchCV(model_gbdt, param_grid, cv=5,scoring='roc_auc')
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


### 不添加情绪
from sklearn.ensemble import GradientBoostingClassifier
param_grid = {
              "n_estimators": [10, 50,100,200]
             }
model_gbdt = GradientBoostingClassifier(learning_rate=0.1, random_state=0)
grid_search = GridSearchCV(model_gbdt, param_grid, cv=5,scoring='roc_auc')
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
model_svm = svm.SVC(random_state=0)
grid_search = GridSearchCV(model_svm, param_grid, cv=5,scoring='roc_auc')
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
model_svm = svm.SVC(random_state=0)
grid_search = GridSearchCV(model_svm, param_grid, cv=5,scoring='roc_auc')
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
model_svm = svm.SVC(random_state=0)
grid_search = GridSearchCV(model_svm, param_grid, cv=5,scoring='roc_auc')
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



from sklearn.neural_network import MLPClassifier
param_grid = {'max_iter':[50],'batch_size':[10,50]}
model_mlp = MLPClassifier(hidden_layer_sizes=(30,10), activation='relu', solver='adam', learning_rate_init=0.01,random_state=0)
grid_search = GridSearchCV(model_mlp, param_grid, cv=5,scoring='roc_auc')
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
from sklearn.neural_network import MLPClassifier
param_grid = {'max_iter':[50],'batch_size':[10,50]}
model_mlp = MLPClassifier(hidden_layer_sizes=(30,10), activation='relu', solver='adam', learning_rate_init=0.01,random_state=0)
grid_search = GridSearchCV(model_mlp, param_grid, cv=5,scoring='roc_auc')
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
from sklearn.neural_network import MLPClassifier
param_grid = {'max_iter':[50],'batch_size':[10,50]}
model_mlp = MLPClassifier(hidden_layer_sizes=(30,10), activation='relu', solver='adam', learning_rate_init=0.01,random_state=0)
grid_search = GridSearchCV(model_mlp, param_grid, cv=5,scoring='roc_auc')
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


# ## 5.8 上采样构建平衡数据



train_data_OverSampler = pd.read_csv('train_data001_oversampler.csv').drop('profile',axis=1)
X_ros, y_ros = train_data_OverSampler.drop('funding',axis=1),train_data_OverSampler['funding']



print(y_train.value_counts())
print(y_ros.value_counts())


# ### 5.8.1 决策树

model_dt = DecisionTreeClassifier(max_depth=5,random_state=0)
model_dt.fit(X_ros, y_ros)
print('#################################')
print('Result on test set:')
y_test_pred = model_dt.predict(X=X_test) 
result_dt = result_metrics(y_test, y_test_pred)
print('#################################')


# ### 5.8.2 随机森林

model_rf = RandomForestClassifier(max_depth=10,n_estimators=100,random_state=0)
model_rf.fit(X_ros, y_ros)
print('#################################')
print('Result on test set:')
y_test_pred = model_rf.predict(X=X_test) 
result_rf = result_metrics(y_test, y_test_pred)
print('#################################')


# ### 5.8.3 AdaBoost


dt = DecisionTreeClassifier(max_depth=1, random_state=0)
model_ab = AdaBoostClassifier(base_estimator=dt, n_estimators=50, random_state=0)
model_ab.fit(X_ros, y_ros)
print('#################################')
print('Result on test set:')
y_test_pred = model_ab.predict(X=X_test) 
result_logistic = result_metrics(y_test, y_test_pred)
print('#################################')


# ### 5.8.4 GBDT


model_gbdt = GradientBoostingClassifier(loss='exponential', learning_rate=0.1, n_estimators=100)
model_gbdt.fit(X_ros, y_ros)
print('#################################')
print('Result on test set:')
y_test_pred = model_gbdt.predict(X=X_test) 
result_logistic = result_metrics(y_test, y_test_pred)
print('#################################')


# ### 5.8.5 SVM


model_svm = svm.SVC(kernel='rbf', random_state=0)
model_svm.fit(X_ros, y_ros)
print('#################################')
print('Result on test set:')
y_test_pred = model_svm.predict(X=X_test) 
result_logistic = result_metrics(y_test, y_test_pred)
print('#################################')


# ### 5.8.6 MLP


model_mlp = MLPClassifier(hidden_layer_sizes=(30,10), activation='relu', solver='adam', max_iter=50, batch_size=10,learning_rate_init=0.01,random_state=0)

model_mlp.fit(X_ros, y_ros)
print('#################################')
print('Result on test set:')
y_test_pred = model_mlp.predict(X=X_test) 
result_logistic = result_metrics(y_test, y_test_pred)
print('#################################')

# ## 5.9 下采样构建平衡数据


train_data_UnderSampler = pd.read_csv('train_data001_undersampler.csv').drop('profile',axis=1)
X_ros, y_ros = train_data_UnderSampler.drop('funding',axis=1),train_data_UnderSampler['funding']


model_dt = DecisionTreeClassifier(max_depth=5,random_state=0)
model_dt.fit(X_ros, y_ros)
print('#################################')
print('Result on test set:')
y_test_pred = model_dt.predict(X=X_test) 
result_dt = result_metrics(y_test, y_test_pred)
print('#################################')

model_rf = RandomForestClassifier(max_depth=10,n_estimators=100,random_state=0)
model_rf.fit(X_ros, y_ros)
print('#################################')
print('Result on test set:')
y_test_pred = model_rf.predict(X=X_test) 
result_rf = result_metrics(y_test, y_test_pred)
print('#################################')

dt = DecisionTreeClassifier(max_depth=1, random_state=0)
model_ab = AdaBoostClassifier(base_estimator=dt, n_estimators=50, random_state=0)
model_ab.fit(X_ros, y_ros)
print('#################################')
print('Result on test set:')
y_test_pred = model_ab.predict(X=X_test) 
result_logistic = result_metrics(y_test, y_test_pred)
print('#################################')

model_gbdt = GradientBoostingClassifier(loss='exponential', learning_rate=0.1, n_estimators=100)
model_gbdt.fit(X_ros, y_ros)
print('#################################')
print('Result on test set:')
y_test_pred = model_gbdt.predict(X=X_test) 
result_logistic = result_metrics(y_test, y_test_pred)
print('#################################')

model_svm = svm.SVC(kernel='rbf', random_state=0)
model_svm.fit(X_ros, y_ros)
print('#################################')
print('Result on test set:')
y_test_pred = model_svm.predict(X=X_test) 
result_logistic = result_metrics(y_test, y_test_pred)
print('#################################')

model_mlp = MLPClassifier(hidden_layer_sizes=(30,10), activation='relu', solver='adam', max_iter=50, batch_size=10,learning_rate_init=0.01,random_state=0)

model_mlp.fit(X_ros, y_ros)
print('#################################')
print('Result on test set:')
y_test_pred = model_mlp.predict(X=X_test) 
result_logistic = result_metrics(y_test, y_test_pred)
print('#################################')
