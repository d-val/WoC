import sys
import pandas as pd
import numpy as np
from operator import itemgetter
from sklearn import preprocessing
import scipy.stats.mstats as sp
import datetime as DT
from time import time
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.rinterface import parse
# import rpy2.robjects.lib.ggplot2 as ggplot2

from rpy2.robjects.conversion import localconverter

import urllib,json,datetime, copy,time

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def dl_futures_data(startDate, endDate, instrument):
	write_csv = False
	fetch_online = True
	if fetch_online:
		futures_data = pd.DataFrame()
		url = "http://ondemand.websol.barchart.com/getHistory.json?symbol=%s&apikey=ondemand&type=daily&startDate=%s&endDate=%s&order=asc" %(instrument, startDate, endDate)
		response = urllib.urlopen(url)
		data = json.loads(response.read())

		for row in data['results']:
			futures_data = futures_data.append(row, ignore_index=True)

		if write_csv == True:
			futures_data.to_csv('data/futures_data'+str(datetime.datetime.now())+'.csv')
	else:
		futures_data = pd.read_csv('data/futures_data2016-07-14 23:39:31.022659.csv')
	return futures_data


def getTrueValues():
	end = ['2016-06-24', '2016-07-15', '2016-08-05','2016-10-07','2016-10-28','2016-12-02','2016-12-19']
	instru = ['ESY00', 'CLY00', 'GCQ16','ESY00','ESY00','ESY00','ESY00']
	trueValue = [2037.4100000000001, 45.939999999999998, 1336.4000000000001, 2153.7399999999998, 2126.4099999999999, 2191.9499999999998, 2262.5300000000002]
	return trueValue


def bootstrap(df):
	N=len(df)
	df2=df.iloc[np.random.randint(N, size=N)]
	return df2

def readData(rounds_real_name, rd):
	start = ['2016-06-01', '2016-06-28', '2016-07-16', '2016-09-17', '2016-10-15', '2016-11-12',
			 '2016-11-28']  
	deadline = ['2016-06-17', '2016-07-08', '2016-07-29', '2016-09-30', '2016-10-21', '2016-11-25',
				'2016-12-12']  
	end = ['2016-06-24', '2016-07-15', '2016-08-05', '2016-10-07', '2016-10-28', '2016-12-02',
		   '2016-12-19']  
	instru = ['ESY00', 'CLY00', 'GCY00', 'ESY00', 'ESY00', 'ESY00', 'ESY00']

	# print('reading file: ', "data/Data_Round_fixed{}.csv".format(rounds_real_name))
	# df=pd.read_csv("data/Data_Round_fixed{}.csv".format(rounds_real_name))
	df=pd.read_csv("raw data/Data_Round_{}.csv".format(rounds_real_name), engine='python')

	# df['created_on'] = pd.to_datetime(df['created_on'], format='%Y-%m-%d')
	# df = df[ df['created_on'] <= pd.to_datetime( deadline[int(rd-1)] , format='%Y-%m-%d')]
	# print(df)

	df['log_target_price_after'] = np.log(df.target_price)
	df['log_crowd_mean'] =  np.log(df.CrowdMean)


	df['log_market6mmean'] = np.log(df.conjugateCrowdMarket6MPredicted_Mean)
	# df['log_market3wmean'] = np.log(df.conjugateCrowdMarket3WPredicted_Mean)

	df['ESI'] = abs(df.log_target_price_after - df.log_crowd_mean) /  df.log_target_price_after
	df['ERP_6'] = abs(df.log_target_price_after - df.log_market6mmean) / df.log_target_price_after
	# df['ERP_3'] = abs(df.log_target_price_after - df.log_market3wmean) / df.log_target_price_after

	assert False != np.any (df.ESI>0)
	# assert False != np.any (df.ERP_3>0)
	assert False != np.any (df.ERP_6>0)

	df['modelDeviations_6']= -df.ESI + df.ERP_6
	# df['modelDeviations_3']= -df.ESI + df.ERP_3

	x = df['modelDeviations_6'].tolist()
	x = [i for i in x if i is not np.nan]
	x = [i for i in x if str(i) != 'nan']

	# print('min max', min(x), max(x))
	# n, bins, patches = plt.hist(x , 50, normed=1, facecolor='green', alpha=0.75)



	return df


def compareWOTC_cum(weighttype,dfo,trueValue,rn, numchunks):
	df=copy.deepcopy(dfo)
	noFilterValues_before=[]
	noFilterValues_after=[]
	for rd in [1,2,3,4,5,6,7]:
		dframe=df[df['round_id']==rd]
		print(dframe.shape)
		noFilterValues_before.append(dframe['pre'].mean())
		noFilterValues_after.append(dframe['target_price'].mean())

	print('noFilterValues_before',  noFilterValues_before)


	df = df[np.isfinite(df[weighttype])]

	ra=[min(df[weighttype]),max(df[weighttype])]
	print('ra range', ra)

	# with localconverter(ro.default_converter + pandas2ri.converter):
 #  		r_from_pd_df = ro.conversion.py2rpy(df)

	# ro.globalenv['r_output'] = r_from_pd_df
	# # ro.globalenv['filename_hist'] = 'modelDeviations_6_hist.png'
	# plot_filename='modelDeviations_6_hist.png'

	
	# ro.r('''
	# 	library('ggplot2')
	# 	p2 <- ggplot(r_output, aes(x=modelDeviations_6)) + geom_histogram() + scale_x_log10() 
	# 	ggsave(file=paste('', "%s", sep=''), width = 5, height = 5, dpi = 300)


	# 	'''%plot_filename
	# 	)



	unique_alphas = pd.unique(df[weighttype])
	unique_alphas.sort()
	alpha_chunks = chunkIt(unique_alphas, numchunks)
	alpharange = []

	for chunk in alpha_chunks:
		alpharange.append(chunk[0])

	print('alpharange', alpharange, len(alpharange))



	alphas=[]
	meanErrors=[]	
	meanErrors_a=[]
	meanErrors_imp=[]
	meanErrors_imp_a=[]
	meanErrors_imp_all=[]
	for alpha in alpharange[1:]:
		if alpha < 0: continue
		sdf=df[(df[weighttype]>0)&(df[weighttype]<=alpha)]
		print('positive alpha only', alpha, sdf.shape)
		roundDf=sdf.groupby('round_id')
		meanSet=[]
		meanSet_a=[]
		meanSet_imp=[]
		meanSet_imp_a=[]
		meanSet_imp_all=[]
		for j in range(rn):
			meanErrorsPG=[]
			meanErrorsPG_a=[]
			meanErrorsPG_imp=[]
			meanErrorsPG_imp_a=[]
			meanErrorsPG_imp_all=[]
			for name,sgroup in roundDf:
				group=bootstrap(sgroup)
				meanValue=group['pre'].mean()
				meanValue_a=group['target_price'].mean()
				meanValue_all=(meanValue+meanValue_a)/2
				tv=trueValue[int(group.iloc[0]['round_id'] - 1.0)]
				noFilterValue=noFilterValues_before[int(group.iloc[0]['round_id'] - 1.0)]
				noFilterValue_a=noFilterValues_after[int(group.iloc[0]['round_id'] - 1.0)]
				noFilterValue_all=(noFilterValue + noFilterValue_a)/2

				noFilterError=abs((noFilterValue-tv)/tv)*100.0
				noFilterError_a=abs((noFilterValue_a-tv)/tv)*100.0
				noFilterError_all=abs((noFilterValue_all-tv)/tv)*100.0

				meanErrorsPG.append(abs((meanValue-tv)/tv)*100.0)
				meanErrorsPG_a.append(abs((meanValue_a-tv)/tv)*100.0)
				meanErrorsPG_imp_all.append((noFilterError_all-abs((meanValue_all-tv)/tv)*100.0))
				meanErrorsPG_imp.append((noFilterError-abs((meanValue-tv)/tv)*100.0))
				meanErrorsPG_imp_a.append((noFilterError_a-abs((meanValue_a-tv)/tv)*100.0))
			meanSet.append(np.mean(meanErrorsPG))
			meanSet_a.append(np.mean(meanErrorsPG_a))
			meanSet_imp.append(np.mean(meanErrorsPG_imp))
			meanSet_imp_a.append(np.mean(meanErrorsPG_imp_a))
			meanSet_imp_all.append(np.mean(meanErrorsPG_imp_all))
		alphas.append(round(alpha,20))
		meanErrors.append(meanSet)
		meanErrors_a.append(meanSet_a)
		meanErrors_imp.append(meanSet_imp)
		meanErrors_imp_a.append(meanSet_imp_a)
		meanErrors_imp_all.append(meanSet_imp_all)
	boxDfs=[]
	for i in range(len(alphas)):
		boxDf=pd.DataFrame()
		boxDf['meanErrors']=meanErrors[i]
		boxDf['meanErrors_a']=meanErrors_a[i]
		boxDf['meanErrors_imp']=meanErrors_imp[i]
		boxDf['meanErrors_imp_a']=meanErrors_imp_a[i]
		boxDf['meanErrors_imp_all']=meanErrors_imp_all[i]
		boxDf['idx']=[alphas[i] for j in range(len(meanErrors[i]))]
		boxDfs.append(boxDf)
	cboxDf=pd.concat(boxDfs)
	cboxDf.index=range(len(cboxDf))

	ra=[min(df[weighttype]),0]
	print('ra', ra)


	alphas=[]
	meanErrors=[]
	meanErrors_a=[]
	meanErrors_imp=[]
	meanErrors_imp_a=[]
	meanErrors_imp_all=[]
	for alpha in alpharange[:-1]:
		if alpha > 0: continue
		sdf=df[(df[weighttype]>alpha)&(df[weighttype]<0)]
		print('negative alpha only', alpha, sdf.shape)
		roundDf=sdf.groupby('round_id')
		meanSet=[]
		meanSet_a=[]
		meanSet_imp=[]
		meanSet_imp_a=[]
		meanSet_imp_all=[]
		for j in range(rn):
			meanErrorsPG=[]
			meanErrorsPG_a=[]
			meanErrorsPG_imp=[]
			meanErrorsPG_imp_a=[]
			meanErrorsPG_imp_all=[]
			for name,sgroup in roundDf:
				group=bootstrap(sgroup)
				meanValue=group['pre'].mean()
				meanValue_a=group['target_price'].mean()
				meanValue_all=(meanValue+meanValue_a)/2
				tv=trueValue[int(group.iloc[0]['round_id'] - 1.0)]
				noFilterValue=noFilterValues_before[int(group.iloc[0]['round_id'] - 1.0)]
				noFilterValue_a=noFilterValues_after[int(group.iloc[0]['round_id'] - 1.0)]
				noFilterValue_all=(noFilterValue + noFilterValue_a)/2

				noFilterError=abs((noFilterValue-tv)/tv)*100.0
				noFilterError_a=abs((noFilterValue_a-tv)/tv)*100.0
				noFilterError_all=abs((noFilterValue_all-tv)/tv)*100.0

				meanErrorsPG.append(abs((meanValue-tv)/tv)*100.0)
				meanErrorsPG_a.append(abs((meanValue_a-tv)/tv)*100.0)
				meanErrorsPG_imp_all.append((noFilterError_all-abs((meanValue_all-tv)/tv)*100.0))
				meanErrorsPG_imp.append((noFilterError-abs((meanValue-tv)/tv)*100.0))
				meanErrorsPG_imp_a.append((noFilterError_a-abs((meanValue_a-tv)/tv)*100.0))
			meanSet.append(np.mean(meanErrorsPG))
			meanSet_a.append(np.mean(meanErrorsPG_a))
			meanSet_imp.append(np.mean(meanErrorsPG_imp))
			meanSet_imp_a.append(np.mean(meanErrorsPG_imp_a))
			meanSet_imp_all.append(np.mean(meanErrorsPG_imp_all))
		alphas.append(round(alpha,20))
		meanErrors.append(meanSet)
		meanErrors_a.append(meanSet_a)
		meanErrors_imp.append(meanSet_imp)
		meanErrors_imp_a.append(meanSet_imp_a)
		meanErrors_imp_all.append(meanSet_imp_all)
	nboxDfs=[]
	for i in range(len(alphas)):
		boxDf=pd.DataFrame()
		boxDf['meanErrors']=meanErrors[i]
		boxDf['meanErrors_a']=meanErrors_a[i]
		boxDf['meanErrors_imp']=meanErrors_imp[i]
		boxDf['meanErrors_imp_a']=meanErrors_imp_a[i]
		boxDf['meanErrors_imp_all']=meanErrors_imp_all[i]
		boxDf['idx']=[alphas[i] for j in range(len(meanErrors[i]))]
		nboxDfs.append(boxDf)
	ncboxDf=pd.concat(nboxDfs)
	ncboxDf.index=range(len(cboxDf),len(cboxDf)+len(ncboxDf))






	# filename_imp_all='wotc_rates_equal_bins/'+weighttype+'performance_improvement_all.png'
	# filename_imp='wotc_rates_equal_bins/'+weighttype+'performance_improvement.png'
	# filename_imp_a='wotc_rates_equal_bins/'+weighttype+'performance_improvement_a_{}.png'.format(numchunks)
	filename_imp_a='plots/'+weighttype+'performance_improvement_a_{}.png'.format(numchunks)

	cboxDfAll=pd.concat([cboxDf,ncboxDf])
	print('pd.unique(cboxDfAll["idx"])', pd.unique(cboxDfAll['idx']), len(pd.unique(cboxDfAll['idx'])), )
	cboxDfAll.to_csv('plots/'+'cboxDfAll_SIvsRealRates.csv', index=False)



	cboxDfAll.index=range(len(cboxDfAll))
	# r.library("grid")
	# rdf =pandas2ri.py2ri(cboxDfAll)
	with localconverter(ro.default_converter + pandas2ri.converter):
  		r_from_pd_df = ro.conversion.py2rpy(cboxDfAll)
	ro.globalenv['r_output'] = r_from_pd_df	
	# ro.globalenv['r_output'] = rdf

	ro.r('''
		library('ggplot2')
		p0<-ggplot(r_output,aes(x=factor(idx),y=meanErrors_imp_a))+ geom_boxplot()
		p0<-p0 + labs(x='Deviation from Bayesian',y='Improvement')+ ggtitle('WOTC comparison')  + scale_fill_manual(values=c("cornflowerblue","red"))
		ggsave(file=paste('',"%s", sep=''), width=8, dpi = 300)
	'''%filename_imp_a) 


	data_agg = cboxDfAll[['meanErrors_imp_a', 'idx']]
	data_agg = data_agg.groupby(['idx']).agg(['mean', 'std', 'count'])
	data_agg.reset_index(inplace=True)
	data_agg.columns = [	'idx', 'mean', 'std', 'count' ]
	data_agg['se'] = data_agg['std']/(data_agg['count']**(1/2))

	normalixed_idx = np.linspace(-1, 1,len(pd.unique(data_agg['idx'])))
	# print('>>>>', normalixed_idx, pd.unique(data_simple['idx']) )

	idx_list = np.sort(pd.unique(data_agg['idx']))
	data_agg['new_idx'] = np.nan
	for i in range(len(idx_list)):
		data_agg.loc[ (data_agg['idx'] == idx_list[i]), 'new_idx' ] = normalixed_idx[i]


	data_agg.to_csv('plots/'+'improvement_simple.csv', index=False)

	with localconverter(ro.default_converter + pandas2ri.converter):
		data_agg_from_pd_df = ro.conversion.py2rpy(data_agg)
	ro.globalenv['r_output'] = data_agg_from_pd_df

	filename_imp_a_simple='plots/'+weighttype+'performance_improvement_a_simple_{}.pdf'.format(numchunks)

	# ro.globalenv['r_output'] = rdf
	ro.r('''
		library('ggplot2')
		library('ggpubr')
		p0 <- ggplot(r_output, aes(x=new_idx, y=mean))+ 
			  # geom_point() +
			  geom_point(col=rgb(0,0,0.5,0.5)) +
			  # geom_smooth() +
			  geom_smooth(method=lm, se=FALSE, linetype = "dashed") +
			  geom_errorbar(aes(
                    ymin = mean-1.96*se, 
                    ymax = mean+1.96*se
                ), alpha = 0.5) + 
            theme_pubr()
            
		ggsave(file="%s", width = 5, height = 5, dpi = 300)
	'''%filename_imp_a_simple)




if __name__ == '__main__':
	trueValue=getTrueValues()
	weighttypes=['modelDeviations_6']

	rn=100
	dfs=[]
	rounds_real_name={1:1,2:2,3:3,4:7,5:8,6:9,7:12}
	# rounds_real_name={1:1,2:2,3:3,4:7,5:8,6:9,7:12}

	rounds = range(1,8) 

	# numchunks = 32
	numchunks = sys.argv[1]

	for rd in rounds:
		df=readData(rounds_real_name[rd], rd)
		dfs.append(df)
		# break

	dfs=pd.concat(dfs)
	for weighttype in weighttypes:
		compareWOTC_cum(weighttype,dfs,trueValue,rn, numchunks)


