

from __future__ import generators
import urllib, json, logging, time, pprint, sys, psycopg2.extras, psycopg2, datetime, operator
import pandas as pd, pandas.io.sql as pdsql, numpy as np
import matplotlib.pyplot as plt
import readline
import rpy2.robjects as ro
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
pandas2ri.activate()
r.library("ggplot2")
r.library("ggpubr")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler('dashboard.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def dl_futures_data(startDate, endDate, futures):
	if fetch_online:	
		futures_data = pd.DataFrame()
		url = "http://ondemand.websol.barchart.com/getHistory.json?symbol=%s&apikey=<your own API>&type=daily&startDate=%s&endDate=%s&order=asc" %(futures, startDate, endDate)
		logger.info(futures+'Downloading futures data using:'+url)
		response = urllib.urlopen(url)
		data = json.loads(response.read())
		logger.info('Downloaded futures data status: '+ data['status']['message'])
		for row in data['results']:
			futures_data = futures_data.append(row, ignore_index=True)
		logger.info('Size futures data fetched '+str(futures_data.shape))
		if write_csv == True:
			futures_data.to_csv('data/'+futures+'_futures_data.csv')
	else:
		futures_data = pd.read_csv('data/'+futures+'_futures_data.csv')
		logger.info('local read '+str(futures_data.shape))
	return futures_data

def data_merger(weight_df, TRUE_PRICE, query_string, filename, startDate, endDate, futures, instrument):
	recommendation_data = dl_postgresql_data(query_string)	
	recommendation_data = recommendation_data[['target_price', 'created_on']]
	recommendation_data['created_on'] = recommendation_data['created_on'].astype(str).str[:10]
	recommendation_data['simple_date'] = pd.to_datetime(recommendation_data['created_on'], format='%Y-%m-%d')

	recommendation_data = recommendation_data[
		( recommendation_data['simple_date'] >= pd.to_datetime( startDate , format='%Y-%m-%d') )
		&
		( recommendation_data['simple_date'] <= pd.to_datetime( endDate , format='%Y-%m-%d') )
	]
	recommendation_data['simple_date'] = recommendation_data['simple_date'].astype(str)
	recommendation_data = recommendation_data[['target_price', 'simple_date']]


	recommendation_data['day'] = recommendation_data['simple_date'].astype(str).str[8:10].astype(int)

	cum_processed_means = pd.DataFrame()
	for day in np.unique(recommendation_data[['day']]):  
		all_predictions = recommendation_data.loc[(recommendation_data.day <= day)]
		day_mean = np.mean(all_predictions[['target_price']])[0]
		day_sd   = np.std(all_predictions[['target_price']])[0] / np.sqrt(len(all_predictions[['target_price']]))
		cum_processed_means = cum_processed_means.append({
											'day': day,
											'cum_pred_mean' : day_mean,
											'cum_pred_se' : day_sd
											}, ignore_index=True)

	cum_processed_means['crowd_cum_real_error'] =   100.0*(cum_processed_means['cum_pred_mean']-TRUE_PRICE) / TRUE_PRICE
	print cum_processed_means

	futures_data = dl_futures_data(startDate, endDate, futures)
	futures_data = futures_data[['close', 'high', 'low','tradingDay']]
	futures_data['day'] = futures_data['tradingDay'].astype(str).str[8:10].astype(int)
	futures_data = futures_data[futures_data['day'] >= 8]

	real_price_data = dl_futures_data(startDate, endDate, instrument)
	real_price_data = real_price_data[['close', 'high', 'low', 'tradingDay']]
	real_price_data['day'] = real_price_data['tradingDay'].astype(str).str[8:10].astype(int)
	real_price_data = real_price_data[real_price_data['day'] >= 8]
	print real_price_data


	return cum_processed_means, futures_data, real_price_data



def plot_line(weight_df, TRUE_PRICE, filename,  query_string = '', startDate = '', endDate = '', futures =  '', instrument = ''):
	cum_processed_means, futures_data, real_price_data = data_merger(weight_df, 
							TRUE_PRICE,
							query_string, 	
							filename,
							startDate,
							endDate,
							futures,								
							instrument								
							)

	cum_processed_meansrdf = pandas2ri.py2ri(cum_processed_means)
	futures_datardf = pandas2ri.py2ri(futures_data)
	real_price_datardf = pandas2ri.py2ri(real_price_data)
	ro.globalenv['cum_processed_means'] = cum_processed_meansrdf
	ro.globalenv['futures_data'] = futures_datardf
	ro.globalenv['real_price_data'] = real_price_datardf
	ro.globalenv['filename'] = filename
	ro.globalenv['TRUE_PRICE'] = TRUE_PRICE

	ro.r('''
			

			p1 <- 	ggplot() + 
					geom_line(data =cum_processed_means, aes( x = day, y = cum_pred_mean), color = 'red')+
					geom_point(data =cum_processed_means, aes( x = day, y = cum_pred_mean), color = 'red')+
					geom_errorbar(data =cum_processed_means, aes( x = day, y = cum_pred_mean, ymin=cum_pred_mean-cum_pred_se, ymax=cum_pred_mean+cum_pred_se), color = 'red')+

					geom_line(data =futures_data, aes( x = day, y = close), color = 'blue')+
					geom_point(data =futures_data, aes( x = day, y = close), color = 'blue')+
					geom_pointrange(data =futures_data, aes( x = day, y = close, ymin=high, ymax=low), color = 'blue')+
					
					geom_line(data =real_price_data, aes( x = day, y = close), color = 'black')+
					geom_point(data =real_price_data, aes( x = day, y = close), color = 'black')+
					geom_pointrange(data =real_price_data, aes( x = day, y = close, ymin=high, ymax=low), color = 'black')+
					


					
					

					geom_hline(aes(yintercept=TRUE_PRICE), color = 'black', linetype = "dotdash") +
            theme_pubr()+
				
			xlim(8, 24)+
			scale_x_continuous(breaks = seq(8, 24, by = 2)) +
		
            theme(
                legend.position = c(0.7, 0.7),
                legend.direction = "vertical",
                legend.background = element_rect(fill=alpha('blue', 0))
                )+					


			ggsave(file=filename, width = 5, height = 4, dpi = 300)    
		''')


write_csv = True
fetch_online = False


plot_line(  
			'', 
			2037.41,
			'brexit_accuracy_line.pdf', 
			'', 									
			'2016-06-07', 							
			'2016-06-24', 							
			'ESU16',									
			'ESY00'									
			)



