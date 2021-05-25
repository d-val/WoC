import pandas as pd
import numpy as np
from operator import itemgetter
from sklearn import preprocessing
import scipy.stats.mstats as sp
import datetime as DT
from time import time
# from ggplot import *
# import readline
# import rpy2.robjects as ro
# from rpy2.robjects import r
# from rpy2.robjects import pandas2ri
# pandas2ri.activate()
# r.library("ggplot2")
# r.library("ggpubr")

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.rinterface import parse
# import rpy2.robjects.lib.ggplot2 as ggplot2

from rpy2.robjects.conversion import localconverter

import urllib,json,datetime, copy,time
import math
import os
from collections import Counter


current_dir = './'
dirs = os.listdir( current_dir )
all_results = pd.DataFrame()


filename = 'plots/'+'cboxDfAll_SIvsRealRates.csv'
current_result = pd.read_csv(filename)
strategy = filename.split('.')[0]

data_simple = current_result[['meanErrors_imp_a', 'idx']]
normalixed_idx = np.linspace(-1, 1, len(pd.unique(data_simple['idx'])))
print('>>>>', normalixed_idx, pd.unique(data_simple['idx']) )

idx_list = np.sort(pd.unique(data_simple['idx']))
data_simple['new_idx'] = np.nan
for i in range(len(idx_list)):
    data_simple.loc[ (data_simple['idx'] == idx_list[i]),'new_idx' ] = normalixed_idx[i]
data_simple = data_simple[['meanErrors_imp_a', 'new_idx']]


data_agg = data_simple.groupby(['new_idx']).agg(['mean', 'std', 'count'])
data_agg.reset_index(inplace=True)
data_agg.columns = [	'new_idx', 'improvement', 'improvement_std', 'count' ]
data_agg['improvement_se'] = data_agg['improvement_std']/(data_agg['count']**(1/2))


improvement_agg = data_agg[[
'new_idx', 'improvement', 'improvement_std', 'improvement_se'
]]
# data_agg['strategy'] = strategy

# all_results = pd.concat([all_results, data_agg])
# print(all_results.shape)
# print(all_results)


print('analyzing sigma')
filename = 'plots/'+'cboxDfAll_SIvsReal_sigma.csv'
current_result = pd.read_csv(filename)
strategy = filename.split('.')[0]

data_simple = current_result[['meanErrors_imp_a', 'idx']]
print(strategy, np.sort(pd.unique(data_simple['idx'])), len(pd.unique(data_simple['idx'])) )
normalixed_idx = np.linspace(-1, 1,len(pd.unique(data_simple['idx'])))
print('>>>>', normalixed_idx, pd.unique(data_simple['idx']) )

idx_list = np.sort(pd.unique(data_simple['idx']))
data_simple['new_idx'] = np.nan
for i in range(len(idx_list)):
    data_simple.loc[ (data_simple['idx'] == idx_list[i]), 'new_idx' ] = normalixed_idx[i]

data_simple = data_simple[['meanErrors_imp_a', 'new_idx']]

data_agg = data_simple.groupby(['new_idx']).agg(['mean', 'std', 'count'])
data_agg.reset_index(inplace=True)
data_agg.columns = [	'new_idx', 'risk', 'risk_std', 'count' ]
data_agg['risk_se'] = data_agg['risk_std']/(data_agg['count']**(1/2))


risk_agg = data_agg[[
    'new_idx', 'risk', 'risk_std', 'risk_se'
]]


merge_df = improvement_agg.merge(risk_agg,
                                 right_on='new_idx',
                                 left_on='new_idx',
                                 how='inner')

print(merge_df)
# data_agg['strategy'] = strategy

# all_results = pd.concat([all_results, data_agg])
# print(all_results.shape)

# merge_df.to_csv('paretto/data/paretto_data_df_{}.csv'.format(len(pd.unique(merge_df['new_idx']))))
merge_df.to_csv('plots/'+'paretto_data_df_{}.csv'.format(len(pd.unique(merge_df['new_idx']))))


# rdf =pandas2ri.py2ri(data)
# ro.globalenv['r_output'] = rdf

with localconverter(ro.default_converter + pandas2ri.converter):
    rdf = ro.conversion.py2rpy(merge_df)
ro.globalenv['r_output'] = rdf

# plot_filename = 'paretto/paretto_new_{}'.format(len(pd.unique(merge_df['new_idx'])))
# plot_filename = 'paretto_new_{}'.format(len(pd.unique(merge_df['new_idx'])))
plot_filename = 'plots/'+'paretto_new_{}'.format(len(pd.unique(merge_df['new_idx'])))

ro.r('''
        library('ggplot2')
        library('ggpubr')
        p1 <- ggplot(r_output, aes(x=improvement,y = risk, label=new_idx)) +
            geom_smooth(method=lm, se=FALSE, linetype = "dashed") +
            
            geom_point(col=rgb(0,0,0.5,0.5)) +
            geom_errorbar(aes(
                    ymin = risk-1.96*risk_se, 
                    ymax = risk+1.96*risk_se
                ), alpha = 0.5) + 
            geom_errorbarh(aes(
                    xmin = improvement-1.96*improvement_se, 
                    xmax = improvement+1.96*improvement_se), 
                alpha = 0.5) +
            # geom_text(             
            # nudge_x = 0.25, nudge_y = 0.25, 
            # check_overlap = T
            # )+            
           
           theme(axis.text.x = element_text(angle = 90, hjust = 1))+

           
           
           
           theme_pubr()+
            theme(
                legend.position = c(0.7, 0.7),
                legend.direction = "vertical",
                legend.background = element_rect(fill=alpha('blue', 0))
                )+


        ggsave(file="%s.pdf", width = 5, height = 5, dpi = 300)

       '''%plot_filename )


