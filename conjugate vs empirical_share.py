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
r.library("scales")
r.library("ggpubr")
r.library("dplyr")

import urllib,json,datetime, copy,time
import math

filename_data = 'learning_dic_all.csv'
crowd_vs_empirical = [
                      "deGrootModel", 
                      "CrowdMean", 
                      "larger_mode", 
                      "CrowdNormal_Crowd_Mean", 
                      "conjugateCrowdMarket6MPredicted_Mean",
                      'Market6MNormal_Market6MPredicted_Mean', 
                      ]

data = pd.read_csv(filename_data)
print(data.shape)

data_filtered = data.loc[data['model'].isin(crowd_vs_empirical)]
print(data_filtered.shape)

data_filtered['mean'] = 100.0*data_filtered['mean']
data_filtered['mean5'] = 100.0*data_filtered['mean5']
data_filtered['mean95'] = 100.0*data_filtered['mean95']
rdf =pandas2ri.py2ri(data_filtered)
ro.globalenv['r_output'] = rdf
ro.globalenv['crowd_vs_empirical'] = crowd_vs_empirical
filename = 'simple_model_comparison'

ro.r('''
        r_output$round <- factor(r_output$round)
        positions <- crowd_vs_empirical
        r_output$model = factor(r_output$model, levels = positions)

        p1 <- ggplot(r_output, aes(x=round,y = mean, fill = model)) +
            geom_bar(stat="identity", color="black", position=position_dodge(), size = 0.1) +
            geom_errorbar(aes(ymin=mean5, ymax=mean95), width=.2, position=position_dodge(.9))+
            theme_pubr() +
           theme(legend.direction="vertical")+
           theme(legend.position="right")+
            theme(axis.line.y = element_line(colour = "black"))+
        ggsave(file="%s.pdf", width = 10, height = 5, dpi = 100)

       '''%filename )































































