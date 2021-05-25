

import pandas as pd
import numpy as np
from pprint import pprint
from collections import Counter
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import linregress
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
global true_value, dic, accuracy, ind_list
import sys, os, scipy.stats
import scipy as sp

def mean_confidence_interval(data, confidence=0.95):
	a = 1.0*np.array(data)
	n = len(a)
	m, se = np.nanmean(a), scipy.stats.sem(a, nan_policy = 'omit')
	h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
	return [m, m-h, m+h, se]

def conjugate(stats, accuracy, dic):
	global dic_all
	print 'working on', stats
	dic['conjugate' + stats] = {}
	accuracy['conjugate' + stats] = {}
	ind_list = [1, 2, 3, 7, 8, 9, 12]
	for ind in ind_list:
		data = pd.read_csv('data/Data_Round_{}.csv'.format(ind))
		data['new'] = 0.5 * (data['target_price'] + data[stats].values)
		data['dif'] = [abs(i - j) / i for i, j in zip(data['target_price_after'].values, data['new'].values)]
		data['acc'] = [abs(true_value[ind_list.index(ind)] - j) / true_value[ind_list.index(ind)] for j in data['new'].values]
		m, m1, m2, sem = mean_confidence_interval(data = data.dif.values)
		dic_all = dic_all.append({
						'model': 'conjugate' + stats,
						'round': ind,
						'mean': m,
						'mean5': m1,
						'mean95': m2, 
						'sem':sem
						}, ignore_index=True)
		dic['conjugate' + stats]['round' + str(ind) + ': mean'] = m
		dic['conjugate' + stats]['round' + str(ind) + ': 5%'] = m1
		dic['conjugate' + stats]['round' + str(ind) + ': 95%'] = m2
		dic['conjugate' + stats]['round' + str(ind) + ': sem'] = sem
		m, m1, m2, sem = mean_confidence_interval(data.acc.values)
		accuracy['conjugate' + stats]['round' + str(ind) + ': mean'] = m
		accuracy['conjugate' + stats]['round' + str(ind) + ': 5%'] = m1
		accuracy['conjugate' + stats]['round' + str(ind) + ': 95%'] = m2
		accuracy['conjugate' + stats]['round' + str(ind) + ': sem'] = sem
	return accuracy, dic

def model_accuracy_mean(accuracy, dic):
	global dic_all
	interested_cols = [
	"CrowdNormal_Crowd_Mean", "CrowdNormal_Crowd_Mode", "CrowdUniform_Crowd_Mean", "CrowdUniform_Crowd_Mode", 
	"Market6MNormal_Market6M_Mean", "Market6MNormal_Market6M_Mode", "deGrootModel", 
	"Market6MUniform_Market6M_Mean", "Market6MUniform_Market6M_Mode", 
	"conjugateCrowdMarket6MPredicted_Mean", "conjugateCrowdMarket6MPredicted_Median", 
	'Market6MNormal_Market6MPredicted_Mean', 'Market6MNormal_Market6MPredicted_Mode', 
	'Market6MUniform_Market6MPredicted_Mean', 'Market6MUniform_Market6MPredicted_Mode']
	for ind in ind_list:
		data = pd.read_csv('data/Data_Round_{}.csv'.format(ind))
		data_fixed = pd.read_csv('datas/Data_Round_fixed{}.csv'.format(ind))
		for replace in ['conjugateCrowdMarket6MPredicted_Median']:
			data[replace] = data_fixed[replace]
		for stats in interested_cols:
			print 'working on ', stats
			data['acc'] = [abs(true_value[ind_list.index(ind)] - j) / true_value[ind_list.index(ind)] for j in data[stats]]
			m, m1, m2, sem = mean_confidence_interval([abs(j-k)/k for j, k in zip(data['target_price_after'].values, data[stats].values)])
			if stats[-5:] == '_Mode':
				name = stats[:-5] + '_Median'
				if name not in dic:
					dic[stats[:-5] + '_Median'] = {}
				dic[stats[:-5] + '_Median']['round' + str(ind) + ': mean'] = m
				dic[stats[:-5] + '_Median']['round' + str(ind) + ': 5%'] = m1
				dic[stats[:-5] + '_Median']['round' + str(ind) + ': 95%'] = m2
				dic[stats[:-5] + '_Median']['round' + str(ind) + ': sem'] = sem
				dic_all = dic_all.append({
								'model': stats[:-5] + '_Median',
								'round': ind,
								'mean': m,
								'mean5': m1,
								'mean95': m2, 
								'sem': sem 
								}, ignore_index=True)
			else:
				if stats not in accuracy:
					dic[stats] = {}
				dic[stats]['round' + str(ind) + ': mean'] = m
				dic[stats]['round' + str(ind) + ': 5%'] = m1
				dic[stats]['round' + str(ind) + ': 95%'] = m2
				dic[stats]['round' + str(ind) +  ': sem'] = sem

				dic_all = dic_all.append({
								'model': stats,
								'round': ind,
								'mean': m,
								'mean5': m1,
								'mean95': m2, 
								'sem': sem
								}, ignore_index=True)

			m, m1, m2, sem= mean_confidence_interval(data.acc.values)
			if stats[-5:] == '_Mode':
				name = stats[:-5] + '_Median'
				if name not in accuracy:
					accuracy[stats[:-5] + '_Median'] = {}
				accuracy[stats[:-5] + '_Median']['round' + str(ind) + ': mean'] = m
				accuracy[stats[:-5] + '_Median']['round' + str(ind) + ': 5%'] = m1
				accuracy[stats[:-5] + '_Median']['round' + str(ind) + ': 95%'] = m2
				accuracy[stats[:-5] + '_Median']['round' + str(ind) + ': sem'] = sem
			else:
				if stats not in accuracy:
					accuracy[stats] = {}
				accuracy[stats]['round' + str(ind) + ': mean'] = m
				accuracy[stats]['round' + str(ind) + ': 5%'] = m1
				accuracy[stats]['round' + str(ind) + ': 95%'] = m2
				accuracy[stats]['round' + str(ind) + ': sem'] = sem
	return accuracy, dic

def conjugate_uni(stats, dic, accuracy):
	global dic_all
	print 'working on ', stats
	dic[stats] = {}
	accuracy[stats] = {}
	ind_list = [1, 2, 3, 7, 8, 9, 12]
	for ind in ind_list:
		data = pd.read_csv('data/Data_Round_{}.csv'.format(ind))
		data_1 = data[data['unimodal'] == True]
		data_1['new'] = 0.5 * (data_1['target_price'] + data_1['mode_11'].values)
		data_1['dif'] = [abs(i - j) / i for i, j in zip(data_1['target_price_after'].values, data_1['new'].values)]
		data_1['acc'] = [abs(true_value[ind_list.index(ind)] - j) / true_value[ind_list.index(ind)] for j in data_1['new'].values]

		data_2 = data[data['unimodal'] == False]
		data_2['new'] = 0.5 * (data_2['target_price'] + data_2[stats].values)
		data_2['dif'] = [abs(i - j) / i for i, j in zip(data_2['target_price_after'].values, data_2['new'].values)]
		data_2['acc'] = [abs(true_value[ind_list.index(ind)] - j) / true_value[ind_list.index(ind)] for j in data_2['new'].values]
		dif = list(data_1['dif'])
		dif.extend(list(data_2['dif']))
		acc = list(data_1['acc'])
		acc.extend(list(data_2['acc']))
		m, m1, m2, sem = mean_confidence_interval(dif)
		dic[stats]['round' + str(ind) + ': mean'] = m
		dic[stats]['round' + str(ind) + ': 5%'] = m1
		dic[stats]['round' + str(ind) + ': 95%'] = m2
		dic[stats]['round' + str(ind) + ': sem'] = sem
		dic_all = dic_all.append({
						'model': stats,
						'round': ind,
						'mean': m,
						'mean5': m1,
						'mean95': m2, 
						'sem': sem
						}, ignore_index=True)
		m, m1, m2, sem = mean_confidence_interval(acc)
		accuracy[stats]['round' + str(ind) + ': mean'] = m
		accuracy[stats]['round' + str(ind) + ': 5%'] = m1
		accuracy[stats]['round' + str(ind) + ': 95%'] = m2
		accuracy[stats]['round' + str(ind) + ': sem'] = sem
	return accuracy, dic

def conjugate_uni_near(accuracy, dic):
	global dic_all
	ind_list = [1, 2, 3, 7, 8, 9, 12]
	stats = 'closer_mode'
	dic[stats] = {}
	accuracy[stats] = {}
	for ind in ind_list:
		data = pd.read_csv('data/Data_Round_{}.csv'.format(ind))

		data_1 = data[data['unimodal'] == True]
		data_1['new'] = 0.5 * (data_1['target_price'] + data_1['mode_11'].values)
		data_1['dif'] = [abs(i - j) / i for i, j in zip(data_1['target_price_after'].values, data_1['new'].values)]
		data_1['acc'] = [abs(true_value[ind_list.index(ind)] - j) / true_value[ind_list.index(ind)] for j in data_1['new'].values]

		data_2 = data[data['unimodal'] == False]
		data_2.reset_index(inplace = True)
		data_2_new = []
		for k in range(len(data_2)):
			if abs(data_2['mode_21'][k] - data_2['target_price'][k]) <=  abs(data_2['mode_22'][k] - data_2['target_price'][k]):
				data_2_new.append(0.5 * (data_2['target_price'][k] + data_2['mode_21'][k]))
			else:
				data_2_new.append(0.5 * (data_2['target_price'][k] + data_2['mode_22'][k]))

		data_2['new'] = data_2_new
		data_2['dif'] = [abs(i - j) / i for i, j in zip(data_2['target_price_after'].values, data_2['new'].values)]
		data_2['acc'] = [abs(true_value[ind_list.index(ind)] - j) / true_value[ind_list.index(ind)] for j in data_2['new'].values]

		dif = list(data_1['dif'])
		dif.extend(list(data_2['dif']))
		acc = list(data_1['acc'])
		acc.extend(list(data_2['acc']))
		m, m1, m2, sem= mean_confidence_interval(dif)
		dic[stats]['round' + str(ind) + ': mean'] = m
		dic[stats]['round' + str(ind) + ': 5%'] = m1
		dic[stats]['round' + str(ind) + ': 95%'] = m2
		dic[stats]['round' + str(ind) + ': sem'] = sem
		dic_all = dic_all.append({
						'model': stats,
						'round': ind,
						'mean': m,
						'mean5': m1,
						'mean95': m2, 
						'sem': sem
						}, ignore_index=True)

		m, m1, m2, sem= mean_confidence_interval(acc)
		accuracy[stats]['round' + str(ind) + ': mean'] = m
		accuracy[stats]['round' + str(ind) + ': 5%'] = m1
		accuracy[stats]['round' + str(ind) + ': 95%'] = m2
		accuracy[stats]['round' + str(ind) + ': sem'] = sem

	return accuracy, dic

def conjugate_aggregate():
	conj_acc, conj_dif = {}, {}
	conj_acc, conj_dif = conjugate_uni('mode_21', conj_acc, conj_dif)
	conj_acc, conj_dif = conjugate_uni('mode_22', conj_acc, conj_dif)
	conj_acc, conj_dif = conjugate_uni_near(conj_acc, conj_dif)
	conj_acc, conj_dif = model_accuracy_mean(conj_acc, conj_dif)
	for stats in ['CrowdMean', 'Market6MMean', 'CrowdMedian', 'Market6MMedian']:
		print 'working on', stats
		conj_acc, conj_dif = conjugate(stats, conj_acc, conj_dif)
		print('>>>>>>>> dic_all.shape', dic_all.shape)
	return conj_acc, conj_dif


true_value = [2037.41, 45.95, 1335.8, 2153.74, 2126.41, 2191.95, 2262.53, 2268.9]
ind_list = [1, 2, 3, 7, 8, 9, 12]
dic_all = pd.DataFrame()
accuracy, dic = conjugate_aggregate()
pd.DataFrame(accuracy).T.to_csv('real_price_accuracy.csv')
pd.DataFrame(dic).T.to_csv('real_price_learning.csv')
dic_all.to_csv('learning_dic_all.csv', index=False)

