import pandas as pd 
import numpy as np 

"""
@author: Xiaolu
@time: 2023/5/1 13:51
"""
# Index(['names', 'vision', 'profile', 'funding', 'size1', 'size2',
#        'founderNum', 'founder_exp', 'markets', 'locations', 'founder_loc'])

# str ='Mobile		Digital Entertainment		Entertainment Industry'
def transfer_list(df_list):
	# items = set()
	items_list = []
	for row in df_list:
		if type(row) == float:
			continue
		words = row.strip().split('\t\t')
		# items.update(words)

		for word in words:
			items_list.append(word)
	items = set(items_list)
	return items,items_list

def tuple_bulid(data_name,data_list,item_num):
	d = pd.DataFrame({data_name:data_list})
	a = pd.DataFrame(d.value_counts())
	row_name = a.index.tolist()
	counts = a[0].tolist()
	data_dict = dict()
	col_name = []
	k = -1
	for name in row_name:
		k += 1
		if k<item_num:
			data_dict[name[0]] = k
			col_name.append(data_name+'_'+name[0])
		else:
			data_dict[name[0]] = item_num
	col_name.append(data_name+'_others')
	return data_dict,col_name


def build_onehot(data_name,col_names,df_list,data_dict,item_num):
	m = len(df_list)
	out_dict = {}


	for i in range(item_num+1):
		out_dict[col_names[i]] = [0 for j in range(m)]

	for i in range(m):
		row = df_list[i]

		if type(row) == float:
			continue
		words = row.strip().split('\t\t')
		for word in words:
			index = data_dict[word]
			if index<item_num:
				out_dict[data_name+'_'+word][i] = 1
			else:
				out_dict[data_name+'_others'][i] = 1

	return out_dict

# w = 
# w = transfer_onehot(data['locations'])


def Bulid_emb(path,market_num,location_num,founderloc_num):
	data = pd.read_excel(path)
	markets,markets_list = transfer_list(data['markets'])
	locations,locations_list = transfer_list(data['locations'])
	founder_loc,founder_loc_list = transfer_list(data['founder_loc'])

	markets_dict,markets_names = tuple_bulid('markets',markets_list,market_num)
	locations_dict,locations_names = tuple_bulid('locations',locations_list,location_num)
	founder_loc_dict,fl_names = tuple_bulid('founder_loc',founder_loc_list,founderloc_num)
	# print(founder_loc_dict)
	# print(fl_names)


	markets_onehot = build_onehot('markets',markets_names,data['markets'],markets_dict,market_num)
	location_onehot = build_onehot('locations',locations_names,data['locations'],locations_dict,location_num)
	founderloc_onehot = build_onehot('founder_loc',fl_names,data['founder_loc'],founder_loc_dict,founderloc_num)
	# print(len(founderloc_onehot['founder_locothers']))
	return markets_onehot,location_onehot,founderloc_onehot
