
import pandas as pd
import numpy as np
from operator import itemgetter
from sklearn import preprocessing
import scipy.stats.mstats as sp
import datetime as DT
from time import time
from ggplot import *
import readline
import readline
import rpy2.robjects as ro
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
import math
pandas2ri.activate()
r.library("ggplot2")
r.library("scales")
r.library("ggthemes")
r.library("ggpubr")

import urllib,json,datetime, copy,time

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

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
	trueValue = [2037.4100000000001, 45.939999999999998, 1336.4000000000001, 2153.7399999999998, 2126.4099999999999, 2191.9499999999998, 2262.5300000000002]

	df=pd.read_csv("data/Data_Round_fixed{}.csv".format(rounds_real_name))

	df['created_on'] = pd.to_datetime(df['created_on'], format='%Y-%m-%d')
	df = df[ df['created_on'] <= pd.to_datetime( deadline[int(rd-1)] , format='%Y-%m-%d')]

	df['post_social'] = (df['target_price_after'] - trueValue[int(rd-1)] ) / trueValue[int(rd-1)] * 100
	df['pre_social']  = (df['target_price']       - trueValue[int(rd-1)] ) / trueValue[int(rd-1)] * 100
	error_post = np.mean(df['post_social'])
	error_pre  = np.mean(df['pre_social'])
	error_post_SE = 1.96*np.std(df['post_social'])/math.sqrt(df['post_social'].shape[0])
	error_pre_SE  = 1.96*np.std(df['pre_social'])/math.sqrt(df['pre_social'].shape[0])

	global dfs
	dfs = dfs.append({
		'round': int(rd-1),
		'error': error_pre,
		'error95' : error_pre + error_pre_SE,
		'error5': error_pre - error_pre_SE,
		'error_type': 'error_before'
	}, ignore_index=True)

	dfs = dfs.append({
		'round': int(rd-1),
		'error': error_post,
		'error95': error_post + error_post_SE,
		'error5': error_post - error_post_SE,
		'error_type': 'error_post'
	}, ignore_index=True)



if __name__ == '__main__':
	trueValue=getTrueValues()

	dfs= pd.DataFrame()
	rounds_real_name={1:1,2:2,3:3,4:7,5:8,6:9,7:12}

	rounds = range(1,8) 

	for rd in rounds:
		readData(rounds_real_name[rd], rd)

	print dfs

	rdf = pandas2ri.py2ri(dfs)
	ro.globalenv['r_output'] = rdf
	filename = 'errorbar'

	ro.r('''
	        r_output$round <- factor(r_output$round)
	        p1 <- ggplot(r_output, aes(x=round,y = error, fill = error_type)) +
	            geom_bar(stat="identity", color="black", position=position_dodge(), size = 0.1) +
	            geom_errorbar(aes(ymin=error5, ymax=error95), width=.2, position=position_dodge(.9))+
	            theme_pubr()+
	            theme(
	                legend.position = c(0.7, 0.7),
	                legend.direction = "vertical",
	                legend.background = element_rect(fill=alpha('blue', 0))
	                )+

			ggsave(file="%s.pdf", width = 10, height = 5, dpi = 300)

	       ''' % filename)



















