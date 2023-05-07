#!/usr/bin/env python
# coding: utf-8

"""
@author: Xiaolu
@time: 2023/5/1 13:51
"""

# # 3.6 数据划分 训练集 测试集划分

##################### 阈值为0.01  ############################

import pandas as pd
import numpy as np
df = pd.read_excel('./data_one_hot.xlsx')
profile = df[['profile']]
data33 = profile['profile'].copy()
data33 = data33.str.replace('\n','')
profile['profile'] = data33
m,n = df.shape


a_Alldata = pd.read_csv('./data_se.csv')
a_Alldata.dropna(inplace = True) 
a_data_Y = a_Alldata['funding']
a_data_Y_b = a_data_Y.copy() 
a_data_Y_b[a_data_Y>0] = 1
a_data_X = a_Alldata.drop(labels=['funding','size1','size2','markets_others','locations_others','founder_loc_others'],axis=1)
a_data_X.columns = a_data_X.columns.str.replace(' ', '_')
data_e = a_data_X[['name_anger', 'name_disgust', 'name_fear', 'name_joy', 'name_neutral',
       'name_sadness', 'name_surprise', 'vision_anger', 'vision_disgust',
       'vision_fear', 'vision_joy', 'vision_neutral', 'vision_sadness',
       'vision_surprise', 'overview_anger', 'overview_disgust',
       'overview_fear', 'overview_joy', 'overview_neutral', 'overview_sadness',
       'overview_surprise']].applymap(lambda x: 0 if x < 0.01 else x)
data_X1 = a_data_X.copy()
data_X1[['name_anger', 'name_disgust', 'name_fear', 'name_joy', 'name_neutral',
       'name_sadness', 'name_surprise', 'vision_anger', 'vision_disgust',
       'vision_fear', 'vision_joy', 'vision_neutral', 'vision_sadness',
       'vision_surprise', 'overview_anger', 'overview_disgust',
       'overview_fear', 'overview_joy', 'overview_neutral', 'overview_sadness',
       'overview_surprise']] = data_e
data_X1=data_X1.drop(['name_neutral','name_anger', 'name_disgust', 'name_fear', 'name_joy', 'name_sadness',
       'name_surprise', 'vision_anger', 'vision_disgust',
       'vision_fear', 'vision_joy', 'vision_neutral', 'vision_sadness',
       'vision_surprise', 'sentiments_names_neg','sentiments_names_pos','sentiments_vision_neg','sentiments_vision_pos'],axis = 1)
data_all_out = pd.concat([a_data_Y_b,data_X1,profile],axis=1)



data_all_out_drop_emo = data_all_out.drop(['overview_anger', 'overview_disgust', 'overview_fear',
       'overview_joy', 'overview_neutral', 'overview_sadness',
       'overview_surprise'],axis=1)


data_all_out_drop_emo.columns



data_all_out_drop_se = data_all_out.drop(['overview_anger', 'overview_disgust', 'overview_fear',
       'overview_joy', 'overview_neutral', 'overview_sadness',
       'overview_surprise'],axis=1)


data_all_out.to_csv('data_se_all_drop_emo.csv',index=False)


train_data = data_all_out.sample(frac = 0.8, random_state=0)
train_data1 = train_data.sample(frac = 0.8, random_state=0)
train_index = sorted(train_data1.index)
valid_data = train_data.drop(train_data1.index)
valid_index = sorted(valid_data.index)
test_data = data_all_out.drop(train_data.index)
test_index = sorted(test_data.index)
print(train_data1.shape,valid_data.shape,test_data.shape)




train_data.to_csv('train_data001.csv',index=False)
test_data.to_csv('test_data001.csv',index=False)




f_train = open('startup001.binary.train','w',encoding='utf-8')
for i in train_index:
    seq = ' '.join(data_all_out.loc[i].astype('str'))+'\n'      
    f_train.write(seq)
f_train.close()

f_valid = open('startup001.binary.dev','w',encoding='utf-8')
for i in valid_index:
    seq = ' '.join(data_all_out.loc[i].astype('str'))+'\n'      
    f_valid.write(seq)
f_valid.close()

f_test = open('startup001.binary.test','w',encoding='utf-8')
for i in test_index:
    seq = ' '.join(data_all_out.loc[i].astype('str'))+'\n'      
    f_test.write(seq)
f_test.close()



from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0, sampling_strategy='auto') 
X_ros, y_ros = ros.fit_resample(train_data.drop('funding',axis=1),train_data['funding'])
train_data_OverSampler = pd.concat([pd.DataFrame(y_ros),X_ros],axis=1)


train_data_OverSampler.to_csv('train_data001_oversampler.csv',index=False)




train_data_OverSampler1 = train_data_OverSampler.sample(frac = 0.8, random_state=0)
train_index_OverSampler = sorted(train_data_OverSampler1.index)
valid_data_OverSampler = train_data_OverSampler.drop(train_data_OverSampler1.index)
valid_index_OverSampler = sorted(valid_data_OverSampler.index)



f_train = open('startup001over.binary.train','w',encoding='utf-8')
for i in train_index_OverSampler:
    seq = ' '.join(train_data_OverSampler.loc[i].astype('str'))+'\n'      
    f_train.write(seq)
f_train.close()

f_valid = open('startup001over.binary.dev','w',encoding='utf-8')
for i in valid_index_OverSampler:
    seq = ' '.join(train_data_OverSampler.loc[i].astype('str'))+'\n'      
    f_valid.write(seq)
f_valid.close()

f_test = open('startup001over.binary.test','w',encoding='utf-8')
for i in test_index:
    seq = ' '.join(data_all_out.loc[i].astype('str'))+'\n'      
    f_test.write(seq)
f_test.close()


from imblearn.under_sampling import RandomUnderSampler
ros = RandomUnderSampler(random_state=0, sampling_strategy='auto') 
X_ros, y_ros = ros.fit_resample(train_data.drop('funding',axis=1),train_data['funding'])
train_data_UnderSampler = pd.concat([pd.DataFrame(y_ros),X_ros],axis=1)
train_data_UnderSampler.to_csv('train_data001_undersampler.csv',index=False)


train_data_UnderSampler1 = train_data_UnderSampler.sample(frac = 0.8, random_state=0)
train_index_UnderSampler = sorted(train_data_UnderSampler1.index)
valid_data_UnderSampler = train_data_UnderSampler.drop(train_data_UnderSampler1.index)
valid_index_UnderSampler = sorted(valid_data_UnderSampler.index)


f_train = open('startup001under.binary.train','w',encoding='utf-8')
for i in train_index_UnderSampler:
    seq = ' '.join(train_data_UnderSampler.loc[i].astype('str'))+'\n'      
    f_train.write(seq)
f_train.close()

f_valid = open('startup001under.binary.dev','w',encoding='utf-8')
for i in valid_index_UnderSampler:
    seq = ' '.join(train_data_UnderSampler.loc[i].astype('str'))+'\n'      
    f_valid.write(seq)
f_valid.close()

f_test = open('startup001under.binary.test','w',encoding='utf-8')
for i in test_index:
    seq = ' '.join(data_all_out.loc[i].astype('str'))+'\n'      
    f_test.write(seq)
f_test.close()







### 只留下有融资金额的公司进行金额预测
a_data_Y_r_c = np.log10(a_data_Y[a_data_Y>0])
a_data_Y_r = np.log10(a_data_Y[a_data_Y>0]).astype('int')+1
data_all_reg = data_all_out[a_data_Y>0]
data_all_reg['funding'] = a_data_Y_r_c





data_all_reg.to_csv('data_se_all_reg001.csv',index=False)




data_all_reg['funding'] = a_data_Y_r
train_data_reg = data_all_reg.sample(frac = 0.8, random_state=0)
train_data1_reg = train_data_reg.sample(frac = 0.8, random_state=0)
train_index_reg = sorted(train_data1_reg.index)
valid_data_reg = train_data_reg.drop(train_data1_reg.index)
valid_index_reg = sorted(valid_data_reg.index)
test_data_reg = data_all_reg.drop(train_data_reg.index)
test_index_reg = sorted(test_data_reg.index)
print(train_data1_reg.shape,valid_data_reg.shape,test_data_reg.shape)





train_data_reg.to_csv('train_data001_reg.csv',index=False)
test_data_reg.to_csv('test_data001_reg.csv',index=False)




f_train_reg = open('startup001_reg.binary.train','w',encoding='utf-8')
for i in train_index_reg:
    seq = ' '.join(data_all_reg.loc[i].astype('str'))+'\n'      
    f_train_reg.write(seq)
f_train_reg.close()

f_valid_reg = open('startup001_reg.binary.dev','w',encoding='utf-8')
for i in valid_index_reg:
    seq = ' '.join(data_all_reg.loc[i].astype('str'))+'\n'      
    f_valid_reg.write(seq)
f_valid_reg.close()

f_test_reg = open('startup001_reg.binary.test','w',encoding='utf-8')
for i in test_index_reg:
    seq = ' '.join(data_all_reg.loc[i].astype('str'))+'\n'      
    f_test_reg.write(seq)
f_test_reg.close()
