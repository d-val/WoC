import pandas as pd
import numpy as np
from operator import itemgetter
from sklearn import preprocessing
import scipy.stats.mstats as sp
import datetime as DT
from time import time
from ggplot import *
import readline
import rpy2.robjects as ro
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
pandas2ri.activate()
r.library("ggplot2")
r.library("ggthemes")
r.library("ggpubr")
import urllib,json,datetime, copy,time
import math
import os


current_dir = './'
dirs = os.listdir( current_dir )
all_results = pd.DataFrame()

for file in os.listdir( current_dir ):
    if not file.endswith('.csv'): continue
    if 'sigma' in file: continue    
    strategy = ''.join(file.split('_')[1:]).split('.')[0]
    current_result = pd.read_csv(current_dir+'/'+file)

    print(current_result.head())


    data_simple = current_result[['meanErrors_imp_a', 'idx']]
    print(strategy, np.sort(pd.unique(data_simple['idx'])), len(pd.unique(data_simple['idx'])) )
    normalixed_idx = np.linspace(-1, 1,len(pd.unique(data_simple['idx'])))

    idx_list = np.sort(pd.unique(data_simple['idx']))
    data_simple['new_idx'] = np.nan
    for i in range(len(idx_list)):
        data_simple.loc[ (data_simple['idx'] == idx_list[i]),'new_idx' ] = normalixed_idx[i]
    data_simple = data_simple[['meanErrors_imp_a', 'new_idx']]


    data_agg = data_simple.groupby(['new_idx']).agg(['mean', 'std', 'count'])
    data_agg.reset_index(inplace=True)
    data_agg.columns = [	'new_idx', 'mean', 'std', 'count' ]

    data_agg['mean_se_h'] = data_agg['mean'] + 1.96*data_agg['std']/np.sqrt(data_agg['count'])
    data_agg['mean_se_l'] = data_agg['mean'] - 1.96*data_agg['std']/np.sqrt(data_agg['count'])
    data_agg['strategy'] = strategy
    print(all_results.shape)
    all_results = pd.concat([all_results, data_agg])

all_results = all_results.dropna()
print(all_results)


data = all_results


rdf =pandas2ri.py2ri(all_results)
ro.globalenv['r_output'] = rdf
plot_filename = 'all_fanplots'


ro.r('''
        p1 <- ggplot(r_output, aes(x=new_idx,y = mean, color = strategy)) +
           geom_line() +
           geom_errorbar(aes(ymin = mean_se_h,ymax = mean_se_l))+
           theme(axis.text.x = element_text(angle = 90, hjust = 1))+
            theme_pubr()+
            theme(
                legend.position = c(0.7, 0.7),
                legend.direction = "vertical",
                legend.background = element_rect(fill=alpha('blue', 0))
                )+

       ggsave(file="%s.pdf", width = 5, height = 5, dpi = 300)

       '''%plot_filename )

