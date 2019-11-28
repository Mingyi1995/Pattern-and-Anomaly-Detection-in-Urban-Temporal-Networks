#!/usr/bin/env python
# coding: utf-8

# In[7]:


#add necessary libraries
import networkx as nx
import pandas as pd
import numpy as np
import os
from sklearn.mixture import GaussianMixture 
from sklearn.decomposition import PCA
import datetime
import operator
import warnings
warnings.filterwarnings('ignore')


# In[8]:


RecordWritingPath = '/Users/hemingyi/Documents/capstone/post/result/'
TransportationDataPath = '/Users/hemingyi/Documents/capstone/post/transportation/output/'
EventDataPath = '/Users/hemingyi/Documents/capstone/post/events/'
comboPath = '/Users/hemingyi/Documents/capstone/combo/'
PostData = '/Users/hemingyi/Documents/capstone/post/'
# dataFile = TransportationDataPath+city+'EdgeYearwiseAggregated.csv'


# In[9]:


def anomalyDetection(y,ncom,pval = 0.2,iterN=20):
    #index of regular (non-outlier points)

    rind = np.array(range(y.shape[0]))
    
    #clustering model
    gm=GaussianMixture(n_components=ncom,n_init=100,max_iter=1000,random_state=0) 
    for i in range(iterN): #iterate
#         print('Iteration {}'.format(i+1))  
        clustering=gm.fit(y[rind,:]) #fit EM clustering model excluding outliers
        l=clustering.score_samples(y) #estimate likelihood for each point
        Lthres=sorted(l)[int(len(l)*pval)] #anomaly threshold
#         print(Lthres)
        rind0=0+rind
        rind=l>Lthres #non-anomalous points
        if all(rind==rind0):
#             print('Convergence in {} iterations'.format(i+1))
            break
    return l < Lthres


# In[6]:


def anomalyDetection(y,ncomp,pval = 0.2,iterN=20):
    #index of regular (non-outlier points)
    #rind=y[:,0]>-10 
    rind = np.array(range(y.shape[0]))
    
    #clustering model
    gm=GaussianMixture(n_components=ncomp,n_init=100,max_iter=1000,random_state=0) 
    for i in range(iterN): #iterate
#         print('Iteration {}'.format(i+1))  
        clustering=gm.fit(y[rind,:]) #fit EM clustering model excluding outliers
        l=clustering.score_samples(y) #estimate likelihood for each point
        Lthres=sorted(l)[int(len(l)*pval)] #anomaly threshold
        rind0=0+rind
        rind=l>Lthres #non-anomalous points
        if all(rind==rind0):
#             print('Convergence in {} iterations'.format(i+1))
            break
    return l < Lthres


# In[90]:


# import events data
def getEvents(EventDataPath,city):
    events_data =EventDataPath+city+'Events.csv'
    df_events = pd.read_csv(events_data, encoding = "ISO-8859-1", parse_dates=['Date'], infer_datetime_format=True)

    # dataframe for events
    df_finalEvents =  df_events[['Date', 'Type']]
    
    df_finalEvents['National Holiday'] = False
    df_finalEvents['Extreme Weather'] = False
    df_finalEvents['Culture Event'] = False
    df_finalEvents.loc[df_finalEvents['Type'] == 'National Holiday', 'National Holiday'] = True
    df_finalEvents.loc[df_finalEvents['Type'] == 'Extreme Weather', 'Extreme Weather'] = True
    df_finalEvents.loc[df_finalEvents['Type'] == 'Culture Event', 'Culture Event'] = True

    df_finalEvents = df_finalEvents.groupby(['Date']).sum()
    df_finalEvents['Anomaly'] = True
    df_finalEvents.reset_index(inplace=True)
    
    return df_finalEvents


def AnomalyDetectionPipeline(aggregation, dimension, standardize, city):
    data = pd.read_csv(TransportationDataPath+'%s/%s/%s/%s%s%s.csv'%(aggregation, dimension, standardize, city, aggregation, standardize))
    matrix = data.drop(['date'], axis=1).values
    EventsDF = getEvents(EventDataPath,city)
    date = data.date.to_frame().rename(columns={'date':'Date'})
    threresult = {}
    EventsDF['Date'] = EventsDF['Date'].astype('str')
    date['Date'] = date['Date'].astype('str')
    df = date.merge(EventsDF,on='Date',how='left')
    df = df.fillna(False)
    df.replace([2.0,1.0,0.0],[True, True, False])
    for comp in [1,2,3,4,5]:
        for th in np.arange(0,1,0.01):
        	th = round(th,2)
        	print("\r",'GMM Parameter: %s-%s'%(comp,th),end="",flush=True)
	        outliers = anomalyDetection(matrix,comp,pval = th)
	        df['%s-%s'%(comp,th)] = outliers
    df.to_csv(RecordWritingPath+'AnomalyDetectionResult1006/'+'%s%s%s%s.csv'%(city,aggregation,dimension,standardize),index=False)



# AnomalyDetectionPipeline('Comm', 'PCA', 'Whiten', 'Taipei')
standardize = 'Normalize'
city = input('city: ')
for aggregation in ['Comm','IO']:
    for dimension in ['OriginSize','PCA','AE']:
        print(aggregation, dimension, standardize, city)
        AnomalyDetectionPipeline(aggregation, dimension, standardize, city)


# for standardize in ['Normalize', 'Whiten', 'Both']:
#     for city in ['Taipei', 'NewYork', 'DC']:

#         AnomalyDetectionPipeline('Comm', 'OriginSize', standardize, city)

