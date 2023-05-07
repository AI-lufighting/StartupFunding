#!/usr/bin/env python
# coding: utf-8

"""
@author: Xiaolu
@time: 2023/5/1 13:51
"""
# # 1 研究背景

# 初创企业融资很重要

# # 2 研究目的和研究内容

# 使用线性回归方法探究关键因素，使用机器学习方法构建预测模型。
# - 探究初创企业融资成功的关键因素
# - 构建初创企业融资是否成功的预测模型
# - 构建初创企业融资金额的预测模型

# # 3 数据来源与数据预处理

# 采用八爪鱼采集器，获取angel.co网站的初创公司，共爬取？？？家公司，其中？？家公司获得了融资

import pandas as pd 
import numpy as np

data = pd.read_excel('./full_data.xlsx')
c_name = data.columns

data.head()


print(data.info())

# ## 3.1 提取公司基本信息
# - 该信息在‘基本信息’列，包含公司位置、公司规模、融资总额、公司类型、市场

#获取基本信息
informations = data['基本信息']
word_list = set(['Location','Locations','Company size','Total raised','Company type','Markets','Market'])
Locations,Company_size1s,Company_size2s,Total_raiseds,Company_types,Markets = [],[],[],[],[],[]

for information in informations:
    #初始化一个公司的基本信息
	Location = []
	Company_size1 = np.nan
	Company_size2 = np.nan
	Total_raised = 0
	Company_type = []
	Market = []
    #将该公司信息分成list
	cols = information.strip().split('\n')
	i=0
	while i < len(cols):
		col = cols[i]
		if col == 'Location' or col == 'Locations':
			i = i + 1
			while i < len(cols) and cols[i] not in word_list:
				if cols[i][:4].lower() != 'show':
					Location.append(cols[i])
				i = i + 1
		elif col == 'Company size':
			com_size = cols[i+1].strip(' people').split('-')
			if len(com_size)<2:
				Company_size1 = com_size[0].strip('+')
				Company_size2 = 10000
			else:
				Company_size1 = com_size[0]
				Company_size2 = com_size[1]
			i = i + 2
		elif col == 'Total raised':
			totalraised = cols[i+1].strip('$')
			if totalraised[-1] == 'M':
				totalraised = str(int(float(totalraised.strip('M'))*1000))
			elif totalraised[-1] == 'K':
				totalraised = totalraised[:-1]
			elif totalraised[-1] == 'B':
				totalraised = str(int(float(totalraised.strip('B'))*1000000))
			Total_raised = totalraised
			i = i + 2
		elif col == 'Markets' or col == 'Market':
			i = i + 1
			while i < len(cols) and cols[i] not in word_list:
				Market.append(cols[i])
				i = i + 1
		else:
			i = i + 1
	Locations.append('\t\t'.join(Location))
	Company_size1s.append(Company_size1)
	Company_size2s.append(Company_size2)
	Total_raiseds.append(Total_raised)
	Markets.append('\t\t'.join(Market))


# ## 3.2 获取创始人信息

## 特殊字符 有些字代表这一行不是纯人名
exclusion_word_set = set(['student','work','one','founder', 'years','year', 'sales', 'sale', '-','@', 
'at','and', 'cmo', 'ceo', 'co-founder', 'editor', 'in', 'president', 'cto', 'cpo', 'year', '&', 'co-ceo', 
'employee', 'of','engineer', 'manager', ',', '!', '.', 'cofounder', 'the', 'about', 'leader', 'marketing', 'lead', 
'managing', 'director', 'designer', 'studied', 'with', 'in', 'and','international','months','student','a','to','startup']) 

rongzi_informations = data['融资概况']
founder_informations = data['创始人概述']
m,n = data.shape
## 融资概况里面有些行具有创始人信息
for i in range(m):
	if type(rongzi_informations[i]) == float:
		continue
	cols = rongzi_informations[i].strip().split('\n')
	if cols[0][:7] == 'Founder':
		founder_informations[i] = '\n'.join(cols[2:])

        

data['创始人概述'] = founder_informations


data_out1111 = {'Corporate_Name':data['公司名称'],'Mission_Statement':data['公司宗旨'],'Company_Overview':data['公司概述'],
'General_Information':data['基本信息'],'Founders_Information':data['创始人概述']}

data_frame111 = pd.DataFrame(data_out1111)
data_frame111.to_excel('originalData.xlsx',index=False)


# 获取创始人个数
jishu = 0
founder_counts = []
founder_names = []
for i in range(m):
	count = 0
	founder_name = []
	if type(founder_informations[i]) == float:
		founder_counts.append(count)
		founder_names.append([])
		continue
	founder_informations[i] = founder_informations[i].lower()
	cols = founder_informations[i].strip().split('\n')
	k=-1
	for col in cols:
		k = k + 1
		S = 0
		

		row = col.split(' ')
		row_word = set(row)

		chonghe = row_word&exclusion_word_set #重合行，人名和其他信息重合了需要单独处理
		if k==0 and len(chonghe)>0:
			# print(i,'******')
			founder_counts.append(count)
			founder_names.append([])
			jishu+=1
			break

        
		if len(row)>4:# 如果这一行长度大于4 一般不是人名行
			continue
            
		for symbol in [',',':','@',';']:# 如果有这些符号，不是人名行
			if symbol in col:
				S = 1
		if S ==1:
			continue
            
		if len(chonghe)>0:# 
			pass
		else:
			if col != '':
				count += 1
				founder_name.append(col)
	else:
		# if count==0:
		# 	# pass
		# 	print(i,'***')
		founder_counts.append(count)
		founder_names.append(founder_name)
# print(jishu)




# 获取创始人企业和高校经历次数
founder_xinxi = data['文本']
exp_counts = []
for i in range(m):
	exp_count = []
	if founder_names[i] == []:
		exp_counts.append(1)
		continue
	if type(founder_xinxi[i]) == float:
		exp_counts.append(1)
		continue
	founders = set(founder_names[i])
	rows = founder_xinxi[i].strip().split('\n')
	rows_num = len(rows)
	k = 0
	while k<rows_num:
		kk = 0
		if rows[k].lower() in founders:
			while k<rows_num and rows[k] != 'BACKGROUND':
				k+=1
			while k<rows_num and rows[k].lower() not in founders:
				kk += 1
				k+=1
			if k == rows_num:
				expCount = kk / 4
			else:
				expCount = (kk-1) /4
			exp_count.append(expCount)
		else:
			k+=1
	count_sum = 0
	for cs in exp_count:
		count_sum+=cs
	if len(exp_count)>0:
		exp_counts.append(count_sum/len(exp_count))
	else:
		exp_counts.append(np.nan)


# 获取创始人所处位置
# 去掉识别的国家，这里只去g20
g20 = set([
    'Argentina', 'Australia', 'Brazil', 'Canada', 'China',
    'France', 'Germany', 'India', 'Indonesia', 'Italy',
    'Japan', 'Mexico', 'Russia', 'Saudi Arabia', 'South Africa',
    'South Korea', 'Turkey', 'United Kingdom', 'United States', 'European Union'
])

from geotext import GeoText

def loc_count(seq):
	seq = seq.replace('\n','. ')
	places = GeoText(seq)
	citys = set(places.cities)
	if len(citys) > 0:
		no_g20 = citys-g20
	else:
		no_g20 = citys

	return list(no_g20)

founder_locs_geotext = []
for i in range(m):
	if type(founder_xinxi[i]) == float:
		founder_locs_geotext.append('')
		continue
	loc = loc_count(founder_xinxi[i])
	# founder_locs.append(loc)
	founder_locs_geotext.append('\t\t'.join(loc))


import spacy

nlp = spacy.load("en_core_web_sm")


def loc_count(seq):
	seq = seq.replace('\n','. ')
	doc = nlp(seq)
	citys = set([entity.text for entity in doc.ents if entity.label_ == "GPE"])
	if len(citys) > 0:
		no_g20 = citys-g20
	else:
		no_g20 = citys 
	return list(no_g20)
founder_locs = []
for i in range(m):
	if type(founder_xinxi[i]) == float:
		founder_locs.append('')
		continue
	loc = loc_count(founder_xinxi[i])
	# founder_locs.append(loc)
	founder_locs.append('\t\t'.join(loc))



for i in range(m):
    if founder_locs[i] == '':
        founder_locs[i] = founder_locs_geotext[i]


#['公司名称', '公司宗旨', '公司概述', '基本信息', '融资概况', '创始人概述', '文本']
data_out = {'names':data['公司名称'],'vision':data['公司宗旨'],'profile':data['公司概述'],
'funding':Total_raiseds,'size1':Company_size1s,'size2':Company_size2s,
'founderNum':founder_counts,'founder_exp':exp_counts,'markets':Markets,'locations':Locations,
'founder_loc':founder_locs}

data_frame = pd.DataFrame(data_out)
data_frame.to_excel('data_pred.xlsx',index=False)


data_frame.info()


# ## 3.3 类别数据one hot

# In[19]:


from pre_test import Bulid_emb
path = 'data_pred.xlsx'
data = pd.read_excel('./full_data.xlsx')
c_name = data.columns
# 每种类别数据分布保留前多少，其他都是others
market_num = 20
location_num = 10
founderloc_num = 10
markets_onehot,location_onehot,founderloc_onehot = Bulid_emb(path,market_num,location_num,founderloc_num)


data_out1 = {'names':data['公司名称'],'vision':data['公司宗旨'],'profile':data['公司概述'],
'funding':Total_raiseds,'size1':Company_size1s,'size2':Company_size2s,
'founderNum':founder_counts,'founder_exp':exp_counts}
data_out1.update(markets_onehot)
data_out1.update(location_onehot)
data_out1.update(founderloc_onehot)
df = pd.DataFrame(data_out1)
df1 = df.dropna()

df1.head()

df1.to_excel('data_one_hot.xlsx',index=False)


df1.info()


print(df1.columns)
print(len(df1.columns))


# ## 3.4 情感特征构建
# - 分析公司名称、主旨、描述给人的感觉

import pandas as pd 
import numpy as np
df = pd.read_excel('./data_one_hot.xlsx')
data1 = df['names'].tolist()
data2 = df['vision'].tolist()
data3 = df['profile'].tolist()
m,n = df.shape


# ### 3.4.1  sentiment


## sentiment 计算
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


data_out_reg = {}
col_names = ['neg','pos']
for col in col_names:
	sentiments_names = []
	sentiments_vision = []
	sentiments_profile = []
	for i in range(m):
		score1 = analyzer.polarity_scores(data1[i])
		sentiments_names.append(score1[col])

		score2 = analyzer.polarity_scores(data2[i])
		sentiments_vision.append(score2[col])

		score3 = analyzer.polarity_scores(data3[i])
		sentiments_profile.append(score3[col])

	data_out_col = {'sentiments_names_'+col:sentiments_names,'sentiments_vision_'+col:sentiments_vision,'sentiments_profile_'+col:sentiments_profile}
	# data_out_col = {'sentiments_vision_'+col:sentiments_vision,'sentiments_profile_'+col:sentiments_profile}
	data_out_reg.update(data_out_col)

df2 = df.drop(['names','vision','profile'],axis=1)

df_reg = pd.DataFrame(data_out_reg)
df_reg = pd.concat([df_reg,df2],axis=1)
df_reg.to_excel('data_sentiment.xlsx',index=False)


# ### 3.4.2 emotion

from transformers import AutoTokenizer, AutoModelForSequenceClassification,pipeline
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=r"C:\Users\yuan\.cache\huggingface\hub\models--j-hartmann--emotion-english-distilroberta-base\snapshots\0e1cd914e3d46199ed785853e12b57304e04178b")
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=r"C:\Users\yuan\.cache\huggingface\hub\models--j-hartmann--emotion-english-distilroberta-base\snapshots\0e1cd914e3d46199ed785853e12b57304e04178b")#distilroberta-base")
classifier_emotion = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, top_k=None)

## 按照顺序计算overviews的情感得分
columns=['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
Overviews = data3
k = 0
for Overview in Overviews:
    if k%500==0:
        print(k)    
    Overview = str(Overview)
    emotion = classifier_emotion(Overview)[0]
    emotion = sorted(emotion,key=lambda x:x['label'])
    Anger = emotion[0]['score']
    Disgust = emotion[1]['score']
    Fear = emotion[2]['score']
    Joy = emotion[3]['score']
    Neutral = emotion[4]['score']
    Sadness = emotion[5]['score']
    Surprise = emotion[6]['score']
    
    if k==0:
        emotion_score = [[Anger,Disgust,Fear,Joy,Neutral,Sadness,Surprise]]        
        emotion_df = pd.DataFrame(emotion_score,columns=['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'])
    else:
        emotion_score = [Anger,Disgust,Fear,Joy,Neutral,Sadness,Surprise]
        emotion_df.loc[k] = emotion_score
    k+=1 
       

print(emotion_df.head()) 



emotion_df.to_csv('emotion_7.csv',index=False)


## 按照顺序计算name的情感得分
names = data1
k = 0
for name in names:
    if k%1000==0:
        print(k)    
    Overview = str(name)
    emotion = classifier_emotion(name)[0]
    emotion = sorted(emotion,key=lambda x:x['label'])
    Anger = emotion[0]['score']
    Disgust = emotion[1]['score']
    Fear = emotion[2]['score']
    Joy = emotion[3]['score']
    Neutral = emotion[4]['score']
    Sadness = emotion[5]['score']
    Surprise = emotion[6]['score']
    
    if k==0:
        emotion_score = [[Anger,Disgust,Fear,Joy,Neutral,Sadness,Surprise]]        
        emotion_df = pd.DataFrame(emotion_score,columns=['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'])
    else:
        emotion_score = [Anger,Disgust,Fear,Joy,Neutral,Sadness,Surprise]
        emotion_df.loc[k] = emotion_score
    k+=1 

       

print(emotion_df.head()) 

emotion_df.to_csv('emotion_name_7.csv',index=False)


## 按照顺序计算vision的情感得分
visions = data2
k = 0
for vision in visions:
    if k%2000==0:
        print(k)    
    emotion = classifier_emotion(vision)[0]
    emotion = sorted(emotion,key=lambda x:x['label'])
    Anger = emotion[0]['score']
    Disgust = emotion[1]['score']
    Fear = emotion[2]['score']
    Joy = emotion[3]['score']
    Neutral = emotion[4]['score']
    Sadness = emotion[5]['score']
    Surprise = emotion[6]['score']
    
    if k==0:
        emotion_score = [[Anger,Disgust,Fear,Joy,Neutral,Sadness,Surprise]]        
        emotion_df = pd.DataFrame(emotion_score,columns=['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'])
    else:
        emotion_score = [Anger,Disgust,Fear,Joy,Neutral,Sadness,Surprise]
        emotion_df.loc[k] = emotion_score
    k+=1 

       

print(emotion_df.head()) 



emotion_df.to_csv('emotion_vision_7.csv',index=False)


### 情感合并成一个文件
emotion_name = pd.read_csv('emotion_name_7.csv')
emotion_name.columns = ['name_anger', 'name_disgust', 'name_fear', 'name_joy', 'name_neutral', 'name_sadness', 'name_surprise']
emotion_vision = pd.read_csv('emotion_vision_7.csv')
emotion_vision.columns = ['vision_anger', 'vision_disgust', 'vision_fear', 'vision_joy', 'vision_neutral', 'vision_sadness', 'vision_surprise']
emotion_overview = pd.read_csv('emotion_7.csv')
emotion_overview.columns = ['overview_anger', 'overview_disgust', 'overview_fear', 'overview_joy', 'overview_neutral', 'overview_sadness', 'overview_surprise']
data_sen = pd.read_excel('data_sentiment.xlsx')
data_se = pd.concat([emotion_name,emotion_vision,emotion_overview,data_sen],axis=1)
data_se.to_csv('data_se.csv',index=False)


data_se.head()

