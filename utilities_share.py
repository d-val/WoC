

from __future__ import generators
import scipy, sys
import urllib, json, logging, time, pprint, psycopg2, sys, psycopg2.extras,  datetime, operator
import pandas as pd, pandas.io.sql as pdsql, numpy as np, matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import datetime
from datetime import timedelta
from scipy.stats.mstats import mode, gmean, hmean
from scipy.stats import uniform
from scipy import stats

import pandas as pd
import pandas.io.sql as pdsql

import numpy as np
import matplotlib.pyplot as plt
from pandas.io.json import json_normalize

try:
	import IPython.core.ultratb
except ImportError:
	pass
else:
	import sys
	sys.excepthook = IPython.core.ultratb.ColorTB()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler('dashboard.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def KnuthMorrisPratt(text, pattern):
	pattern = list(pattern)

	shifts = [1] * (len(pattern) + 1)
	shift = 1
	for pos in range(len(pattern)):
		while shift <= pos and pattern[pos] != pattern[pos-shift]:
			shift += shifts[pos-shift]
		shifts[pos+1] = shift

	startPos = 0
	matchLen = 0
	for c in text:
		while matchLen == len(pattern) or \
			  matchLen >= 0 and pattern[matchLen] != c:
			startPos += shifts[matchLen]
			matchLen -= shifts[matchLen]
		matchLen += 1
		if matchLen == len(pattern):
			yield startPos


def dl_futures_data(startDate, endDate, instrument, freq,  write_csv, fetch_online, i):
	if fetch_online:
		futures_data = pd.DataFrame()
		url = "http://ondemand.websol.barchart.com/getHistory.json?symbol=%s&apikey=<put your own key here>&type=%s&startDate=%s&endDate=%s&order=asc" %(instrument, freq, startDate, endDate)
		print(url)
		response = urllib.urlopen(url)
		data = json.loads(response.read())

		print('got data', len(data['results']) )
		futures_data = json_normalize(data['results'])

		print('got data into pandas')
		if write_csv == True:
			futures_data.to_csv('data/'+str(instrument)+'_'+str(i)+'.csv')
	else:
		futures_data = pd.read_csv('data/'+str(instrument)+'_'+str(i)+'.csv')
	return futures_data

def data_merger(weight_df, TRUE_PRICE, query_string, filename, startDate, endDate, instrument):
	recommendation_data = dl_postgresql_data(query_string)

	recommendation_data = recommendation_data[['gs_user_name', 'target_price', 'created_on', 'is_influenced', 'is_expired']]
	recommendation_data['simple_date'] = recommendation_data['created_on'].apply(lambda x: x.strftime('%Y-%m-%d'))

	futures_data = dl_futures_data(startDate, endDate, instrument)
	futures_data = futures_data[['close', 'high', 'low', 'timestamp', 'tradingDay', 'volume']]

	merged_data = pd.merge(	left=recommendation_data,
							right=futures_data,
							left_on='simple_date',
							right_on='tradingDay',
							how = 'left')
	logger.info('Size merged futures + prediction data '+str(merged_data.shape))
	minimal_merged_data = merged_data[['gs_user_name', 'target_price', 'close', 'simple_date']]

	if len(weight_df) > 0:
		minimal_merged_data = pd.merge(	left=minimal_merged_data,
									right=weight_df,
									left_on='gs_user_name',
									right_on='gs_user_name',
									how = 'left')
	else:
		print 'weight 1'
		minimal_merged_data['weight'] = 1
	print('Size merged data pre-aggregation by date '+str(minimal_merged_data.shape))
	minimal_merged_data = minimal_merged_data[['target_price', 'weight', 'simple_date']]


	minimal_merged_data['product_target_price_weight'] = minimal_merged_data['target_price']*minimal_merged_data['weight']

	print('Size merged data pre-aggregation by date '+str(minimal_merged_data.shape))

	comparison_data = minimal_merged_data.groupby(['simple_date']).agg(['sum', 'mean', 'std', 'count'])


	comparison_data.reset_index(inplace=True)
	comparison_data.columns = [	'simple_date',
								'target_price_sum', 'target_price_mean', 'target_price_std', 'target_price_count',
								'weight_sum', 'weight_mean', 'weight_std', 'weight_count',
								'product_target_price_weight_sum', 'product_target_price_weight_mean', 'product_target_price_weight_std', 'product_target_price_weight_count']
	comparison_data['pred_SE'] = comparison_data['target_price_std'] / np.sqrt(comparison_data['target_price_count'])

	comparison_data['wtd_avg'] = comparison_data['product_target_price_weight_sum'] / comparison_data['weight_sum']



	comparison_data=comparison_data.rename(columns = {'target_price_mean':'pred_mean'})
	comparison_data=comparison_data.rename(columns = {'target_price_count':'pred_count'})


	data_plot = comparison_data[['simple_date', 'pred_mean', 'pred_SE', 'wtd_avg', 'product_target_price_weight_sum', 'weight_sum'
								 ]]
	data_plot['day'] = data_plot.index

	simple_date_to_day_map = data_plot[['simple_date', 'day']]
	cumulative_data = pd.merge(	left=minimal_merged_data,
							right=simple_date_to_day_map,
							left_on='simple_date',
							right_on='simple_date',
							how = 'left')
	cum_processed_means_se_data = pd.DataFrame()
	for day in np.unique(cumulative_data[['day']]):  
		all_predictions = cumulative_data.loc[(cumulative_data.day <= day)]
		weighted_predictions = data_plot.loc[(data_plot.day <= day)]

		day_mean = np.mean(all_predictions[['target_price']])[0]
		day_mean_weighted = np.sum(weighted_predictions['product_target_price_weight_sum']) / np.sum(weighted_predictions['weight_sum'])
		day_median = np.median(all_predictions[['target_price']])
		count = all_predictions.shape

		day_se   = np.std(all_predictions[['target_price']])[0] / np.sqrt(len(all_predictions[['target_price']]))
		cum_processed_means_se_data = cum_processed_means_se_data.append({
											'day': day,
											'cum_pred_mean' : day_mean,
											'cum_pred_mean_weighted' : day_mean_weighted,
											'cum_pred_median' : day_median,
											'cum_pred_se' : day_se,
											'count' : count[0]
											}, ignore_index=True)

	data_plot = pd.merge(
						left=data_plot,
						right=cum_processed_means_se_data,
						left_on='day',
						right_on='day',
						how = 'left')
	data_plot['crowd_weighted_error'] =   100.0*(data_plot['cum_pred_mean_weighted']-TRUE_PRICE) / TRUE_PRICE
	data_plot['crowd_real_error'] =   100.0*(data_plot['cum_pred_mean']-TRUE_PRICE) / TRUE_PRICE


	if write_csv:
		data_plot.to_csv('data/'+filename+query_string+'.csv')
	return data_plot


def plot_line(weight_df, TRUE_PRICE, CUT_OFF_DATE, filename,  query = '', startDate = '', endDate = '', instrument = ''):
	data_plot = data_merger(weight_df,
							TRUE_PRICE,
							query, 	
							filename,
							startDate,
							endDate,
							instrument								
							)
	data_plot = data_plot.loc[data_plot['day'] <= 32] 

	print data_plot
	rdf = com.convert_to_r_dataframe(data_plot)
	ro.globalenv['r_output'] = rdf
	ro.globalenv['filename'] = filename+'.png'
	ro.globalenv['TRUE_PRICE'] = TRUE_PRICE
	ro.globalenv['CUT_OFF_DATE'] = CUT_OFF_DATE

	ro.r("""
			p1 <- 	ggplot(r_output) +
					geom_pointrange(aes(	x = day, y = cum_pred_mean, ymin=cum_pred_mean-cum_pred_se, ymax=cum_pred_mean+cum_pred_se), color = 'red', size = .10) +
					geom_text(aes(x = day, y = cum_pred_mean, label=paste(round(crowd_real_error,    2), "%", sep='') ),hjust=1.05, vjust=2,
							 size = 1, color = 'red')+
					geom_hline(aes(yintercept=TRUE_PRICE), color = 'black', linetype = "dotdash") +
					geom_vline(aes(xintercept=CUT_OFF_DATE), color = 'blue', linetype = "dotted", size = 0.5) +

					ylab("Mean Collective Prediction")+
					xlab("Day") +

					theme(panel.background = element_rect(fill = 'grey98', colour = 'grey')) +
					theme(panel.grid.minor = element_line(colour = "grey90"))+
					theme(panel.grid.major = element_line(colour = "grey90"))+

			ggsave(file=filename, width = 5, height = 4, dpi = 300)

		""" )
	return data_plot


def individual_results(true_price, filename, query_string):
	recommendation_data = dl_postgresql_data(query_string)
	results_data = recommendation_data[['mit_recommendation_id',
										'mit_user_id',
										'gs_user_name',
										'created_on',
										'target_price',
										'is_influenced',
										'is_expired'
										]]
	print recommendation_data.shape
	results_data = results_data.sort(['gs_user_name', 'created_on'], ascending = True)

	results_data['deviation (%)'] = (results_data['target_price']/true_price - 1.0)*100.0

	results_data['times'] = results_data.groupby(['gs_user_name']).rank()['created_on']

	results_data.loc[results_data['deviation (%)'] == 0, 'deviation (%)'] = 0.001 
	results_data['precision'] = abs(1.0/results_data['deviation (%)'])
	results_data['frac_progress'] = results_data.apply(deltaT_calc, axis=1)

	results_data['discnt_precision'] = results_data.apply(time_discounter, axis=1)

	results_data = results_data[ (results_data['frac_progress'] < 100) ]

	results_data.to_csv(filename)

def deltaT_calc(df):





	round_start_time = datetime.datetime.strptime('2016-11-28 00:00:00', '%Y-%m-%d %H:%M:%S')
	round_end_time   = datetime.datetime.strptime('2016-12-19 18:00:00', '%Y-%m-%d %H:%M:%S')



	denom = (round_end_time - round_start_time).total_seconds()

	pred_time = df['created_on']
	pred_time_formatted = datetime.datetime.strptime(str(pred_time)[:19], '%Y-%m-%d %H:%M:%S')
	deltaT = (pred_time_formatted-round_start_time).total_seconds()
	if deltaT/denom > 1.0:
		return 100
	else:
		return deltaT/denom

def time_discounter(df):
	precision = df['precision']

	frac_progress = df['frac_progress']
	return precision*np.exp(-5.0*frac_progress)

def scorer(filenameInput, augmentedFilenameInput, userRankOuput):
	precision_data = pd.read_csv(filenameInput)
	user_list = precision_data['gs_user_name'].unique()
	print len(user_list)


	precision_data_augmented_w_histogram_learning = precision_data
	precision_data_augmented_w_histogram_learning['histogram_change'] = None
	user_rank_data = pd.DataFrame()

	for user in user_list:
		user_df = precision_data_augmented_w_histogram_learning.loc[precision_data_augmented_w_histogram_learning['gs_user_name'] == user]
		user_df = user_df.reset_index()


		list_is_influenced = user_df['is_influenced']
		influence_pairs = list(KnuthMorrisPratt(list(list_is_influenced), [False, True]))
		influence_values = []
		for i in influence_pairs:
			histogram_change = abs((user_df.iloc[i+1]['deviation (%)'])/user_df.iloc[i]['deviation (%)'])


			influence_values.append(histogram_change)
			precision_data_augmented_w_histogram_learning.loc[user_df['index'][0]+ i + 1 , 'histogram_change' ] = histogram_change





		last_precision = user_df.iloc[-1]['discnt_precision']
		max_precision  = max( user_df['discnt_precision'])
		mean_precision = np.mean(user_df['discnt_precision'])
		num_pred 	   = user_df.shape[0]

		k1,k2,k3,k4 = 0.4, .1, 0.1, 0.4
		score = 5.0*(k1 * last_precision + \
						k2 * num_pred 					+ \
						k3 * max_precision   + \
						k4 * mean_precision)
		user_rank_data = user_rank_data.append({'gs_user_name': user.strip(), 'score':score, 'histogram_learning': np.mean(influence_values)}, ignore_index=True)

	user_rank_data = user_rank_data.sort(['score'], ascending=[False])
	user_rank_data = user_rank_data.reset_index(drop=True)
	user_rank_data["rank"] = user_rank_data.index.values
	user_rank_data["rank"] = user_rank_data["rank"] + 1
	user_rank_data["rank percent"] = user_rank_data["rank"] / (len(user_list)+1) * 100

	precision_data_augmented_w_histogram_learning.to_csv(augmentedFilenameInput)

	'''
	this takes the true_price and precision_data_augmented_w_histogram_learning, and outputs a score for each user
	the formula of the scoring is TBD but basically:
	???
	- the earlier you predict the better
	- the more accurate you are, the better (+ve, -ve, absolute?)
	- the more you diverge from others?
	- the more you learn from others?
	- the less you are influenced?
	- the more times? you predict the better
	???
	'''

	user_rank_data.to_csv(userRankOuput)
	return user_rank_data

def multiple_round_scores(round1_scores, round2_scores, merged_scoresDF):

	round1_scores = pd.read_csv(round1_scores)
	round2_scores = pd.read_csv(round2_scores)

	round1_scores = round1_scores[['gs_user_name', 'round1_score', 'round2_score', 'round1_rank', 'round2_rank']]
	round2_scores = round2_scores[['gs_user_name', 'histogram_learning', 'score', 'rank']]

	merged_scores = pd.merge(	left=round1_scores,
								right=round2_scores,
								left_on='gs_user_name',
								right_on='gs_user_name',
								how = 'outer'
							)
	merged_scores=merged_scores.rename(columns = {'score':'round3_score'})
	merged_scores=merged_scores.rename(columns = {'rank':'round3_rank'})
	merged_scores = merged_scores.fillna(0)
	merged_scores['sum_score'] = merged_scores['round1_score'] + merged_scores['round2_score'] + merged_scores['round3_score']

	print merged_scores.shape
	merged_scores = merged_scores.sort(['sum_score'], ascending=[False])
	merged_scores = merged_scores.reset_index(drop=True)
	merged_scores["round3_rank_on_sum"] = merged_scores.index.values
	merged_scores["round3_rank_on_sum"] = merged_scores["round3_rank_on_sum"] + 1
	merged_scores["rank percent"] = merged_scores["round3_rank_on_sum"] / (merged_scores.shape[0]+1) * 100
	merged_scores["rank_change"] = merged_scores["round3_rank_on_sum"] - merged_scores["round2_rank"]
	merged_scores.to_csv(merged_scoresDF)


def plotLearningResults():

	'''
		plot the learning results from different models: degroot and bayesian with various distributions
	'''
	sys.stdout = open("result.txt", "w")
	fileName = 'individual_results_Round'
	fileList = [fileName + str(i) + '.csv' for i in [1,2]]
	start = ['2016-06-01', '2016-06-28']
	instru = ['ESY00', 'CLY00']
	for rd in [1, 2]:
		openPrice = dl_futures_data(start[rd-1], time.strftime("%Y-%m-%d"), instru[rd-1]).close[rd - 1]
		result = pd.read_csv('individual_results_Round{}.csv'.format(rd))
		result = result[result['normEmpiricalPostMean']>0]
		result.index = range(len(result))
		influence_pairs = list(KnuthMorrisPratt(list(data.is_influenced), [False, True]))
		influence_pairsPost = [i + 1 for i in influence_pairs]
		result = result[result.index.isin(influence_pairsPost)]
		result.index = range(len(result))
		plt.figure(figsize = (16, 10))
		for i in ['deGroot', 'bayesian', 'normEmpiricalPostMean', 'normEmpiricalPostMode', 'normEmpiricalPostMeanUni', 'normEmpiricalPostModeUni', 'normMiu']:
			deviation = devi(result['target_price'], result[i] , openPrice)
			plt.hist(pd.DataFrame(deviation).dropna(), bins = 50, normed=True, alpha = 0.4, label = i)
			plt.legend()
			plt.savefig('PNGs/histogram_round{}.png'.format(rd))
		print result[['deGroot', 'bayesian', 'normEmpiricalPostMean', 'normEmpiricalPostMode', 'normEmpiricalPostMeanUni', 'normEmpiricalPostModeUni', 'normMiu']].sum()


def posterior(priorMean, priorStd, likelihood, normal = True):
	'''
	prior (mean as input) is either uniform or normal distribution
	Likelihood is the actual histogram
	if normal <- True, prior as normal distribution, if normal = False prior as uniform distribution
	return an estimated posterior histogram
	'''
	likelihood = Counter(likelihood)
	posterior = {}
	normalizeSum = 0
	if normal:
		for k in likelihood:
			posterior[k] = scipy.stats.norm(priorMean, 1).pdf(k) * likelihood[k]
			normalizeSum += posterior[k]
	else:
		for k in likelihood:
			posterior[k] = uniform(priorMean - priorStd, priorMean + priorStd).pdf(k) * likelihood[k]
			normalizeSum += posterior[k]
	for k in posterior.keys():
		posterior[k] = 1.0 * posterior[k]/normalizeSum
	posteriorMean = sum(filter(None, [k * posterior[k] for k in posterior.keys()]))
	posteriorMode = posterior.keys()[np.argmax(posterior.values())]
	return posteriorMean, posteriorMode

def posteriorPriorStdReal(priorMean, priorStd, likelihood, normal = True):
	'''
	prior (mean as input) is either uniform or normal distribution
	Likelihood is the actual histogram
	if normal <- True, prior as normal distribution, if normal = False prior as uniform distribution
	return an estimated posterior histogram
	'''
	likelihood = Counter(likelihood)
	posterior = {}
	normalizeSum = 0
	if normal:
		for k in likelihood:
			posterior[k] = scipy.stats.norm(priorMean, priorStd).pdf(k) * likelihood[k]
			normalizeSum += posterior[k]
	else:
		for k in likelihood:
			posterior[k] = uniform(priorMean - priorStd, priorMean + priorStd).pdf(k) * likelihood[k]
			normalizeSum += posterior[k]
	for k in posterior.keys():
		posterior[k] = 1.0 * posterior[k]/normalizeSum
	posteriorMean = sum(filter(None, [k * posterior[k] for k in posterior.keys()]))
	posteriorMode = posterior.keys()[np.argmax(posterior.values())]
	return posteriorMean, posteriorMode



def normalizeRef(diff, openPrice):
	'''
	diff is the difference in prediction
	return the normalized error (by open Price)
	'''
	return [1.0 *i / openPrice for i in diff]

def devi(actualPrd, estimatedPred, openPrice):
	'''
		normalized deviation
	'''
	return [100.0 * (x - y)/openPrice for x, y in zip(actualPrd, estimatedPred)]

def influenceNew():
	'''
	this function returns :
		1) the characteristics of the histogram the user see
		2) the degroot and bayesian learning results
	'''
	for rd in range(1,4):
		query_string = '''	SELECT mr.*, mi.*
						FROM mit_recommendations mr, mit_user_details mi
						WHERE mi.mit_user_id = mr.mit_user_id
						AND round_id = {}
					'''.format(rd)
		tmp = dl_postgresql_data(query_string)
		print "Prediction data extracted"
		start = ['2016-06-01', '2016-06-28', '2016-07-16']; 
		instru = ['ESY00', 'CLY00', 'GCQ16'] 
		market =  dl_futures_data(start[rd-1], time.strftime("%Y-%m-%d"), instru[rd-1])
		print "Futures data extracted"
		openPrice = market['close'][0]
		stdDic = {str(market['timestamp'][i])[:9]: np.std(market['close'][:i+1]) for i in range(len(market))}
		tmp = tmp.sort(['gs_user_name', 'created_on'], ascending = True)
		pre = [None]
		for i in range(1, len(tmp)):
			pre.append(tmp['target_price'][i-1])
		tmp['pre'] = pre
		wp = 0.5; ws = 1 - wp
		tmp = tmp.sort(['created_on'], ascending = True) 
		hist_mean = [tmp['target_price'][0]]
		hist_mode = [tmp['target_price'][0]]
		hist_median = [tmp['target_price'][0]]
		hist_g_mean = [tmp['target_price'][0]]
		deGroot, bayesian, hist_std = [None], [None], [0]
		normMiuPost= [None]
		normSigmaPost= [None]
		normEmpiricalPostMean = [None]
		normEmpiricalPostMode = [None]
		normEmpiricalPostMeanUni = [None]
		normEmpiricalPostModeUni = [None]
		for i in range(1, len(tmp)):
			try:
				stdPrior = stdDic[str(tmp['created_on'][i] -  timedelta(days=1))[:9]]
			except:
				stdPrior = stdDic[str(tmp['created_on'][i] -  timedelta(days=3))[:9]]
			sampleSize = len(tmp['target_price'][:i])
			hist_mean.append(np.mean(tmp['target_price'][:i])) 
			hist_mode.append(stats.mode(tmp['target_price'][:i])[0][0])
			hist_median.append(np.median(tmp['target_price'][:i]))
			hist_std.append(np.std(tmp['target_price'][:i])) 
			normMiuPost.append((hist_std[-1]**2 * tmp['pre'][i] + sampleSize * stdPrior**2 * hist_mean[-1])/(sampleSize * stdPrior**2 + hist_std[-1]**2)) 
			normSigmaPost.append(np.sqrt((hist_std[-1]**2 * stdPrior**2)/(sampleSize * stdPrior**2 + hist_std[-1]**2)))
			normEmpiricalPostMean.append(posterior(priorMean = tmp['pre'][i], priorStd = stdPrior, likelihood = tmp['target_price'][:i], normal = True)[0])
			normEmpiricalPostMode.append(posterior(priorMean = tmp['pre'][i], priorStd = stdPrior, likelihood = tmp['target_price'][:i], normal = True)[1])
			normEmpiricalPostMeanUni.append(posterior(priorMean = tmp['pre'][i], priorStd = stdPrior, likelihood = tmp['target_price'][:i], normal = False)[0])
			normEmpiricalPostModeUni.append(posterior(priorMean = tmp['pre'][i], priorStd = stdPrior, likelihood = tmp['target_price'][:i], normal = False)[1])
			deGroot.append(0.5 * (hist_mean[-1] + tmp.pre[i]))
			bayesian.append(np.exp(wp * np.log(tmp.pre[i]) + ws * np.log(hist_mean[-1])))

		tmp['hist_mean'] = hist_mean
		tmp['hist_mode'] = hist_mode
		tmp['hist_median'] = hist_median
		tmp['hist_std'] = hist_std
		tmp['normMiu'] = normMiuPost
		tmp['normSigma'] = normSigmaPost
		tmp['normEmpiricalPostMean'] = normEmpiricalPostMean
		tmp['normEmpiricalPostMode'] = normEmpiricalPostMode
		tmp['normEmpiricalPostMeanUni'] = normEmpiricalPostMeanUni
		tmp['normEmpiricalPostModeUni'] = normEmpiricalPostModeUni
		tmp['deGroot'] = deGroot
		tmp['bayesian'] = bayesian
		tmp['devi_degroot'] = tmp['target_price'] - tmp['deGroot']
		tmp['devi_bayesian'] = tmp['target_price'] - tmp['bayesian']
		tmp.to_csv('individual_results_Round{}.csv'.format(rd))

def individualDeviLearning():
	'''
	how much each person's bet deviated from bayesian and deGroot (percentage)
	aggregate over person by taking average, sd, sum
	'''
	for rd in range(1, 4):
		tmp = pd.read_csv('individual_results_Round{}.csv'.format(rd))
		tmp['devi_degroot'] = [(j-i)/i for i, j in zip(tmp['target_price'], tmp['deGroot'])]
		tmp['devi_bayesian'] = [(j-i)/i for i, j in zip(tmp['target_price'], tmp['bayesian'])]
		user, devi_degroot, devi_bayesian , devi_degroot_sum, devi_bayesian_sum, devi_degroot_sd, devi_bayesian_sd = [], [], [], [], [], [], []
		for name, group in tmp.groupby('gs_user_name'):
			user.append(name)
			devi_degroot.append(group['devi_degroot'].mean())
			devi_bayesian.append(group['devi_bayesian'].mean())
			devi_degroot_sum.append(group['devi_degroot'].sum())
			devi_bayesian_sum.append(group['devi_bayesian'].sum())
			devi_degroot_sd.append(group['devi_degroot'].std())
			devi_bayesian_sd.append(group['devi_bayesian'].std())
		print devi_degroot
		plt.figure()
		plt.plot(devi_degroot, '*--', label = 'degroot')
		plt.plot(devi_bayesian, 'o--', label = 'bayesian')
		plt.legend()
		plt.savefig('degroot_bayesian_round{}'.format(rd))

write_csv = False
fetch_online = True
