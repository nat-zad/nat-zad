#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.ticker as mtick
import seaborn as sns


# # The Data
# 
# The data set is comprised of a row per event
# 
# - container journeys included are global
#     - each journey will contain an update record for every time a change in the shipment data is sourced
#     - each update contains all data from prior updates + the new/modified data
#     - updates per [reference_id] will not be a 1-for-1 duplicate and will contain at least 1 change. Updates are not generated if no data has changed since the last time data was sourced for a shipment
# - multimodal events, e.g. vessel, truck, rail, terminal
# - events that have occurred and events estimated to occur are included, see the planned flag
# - vessel and location details included
# - Many events are reported but the focus will be on the 8 core ocean events:
#     - Gate out from origin port
#     - Gate in at origin port
#     - Loaded on vessel at origin port
#     - Vessel departure from origin port
#     - Vessel arrived at destination port
#     - Discharged from vessel at destination port
#     - Gate out from destination port
#     - Gate in empty return

# In[2]:


df = pd.read_csv('Vizion Senior Data Analyst Code Challenge.csv')


# In[3]:


pd.set_option('display.max_columns', None)
df.head(5)


# In[4]:


df.shape


# In[5]:


core = ['Gate out from origin port', 'Gate in at origin port', 'Loaded on vessel at origin port', 'Vessel departure from origin port',
        'Vessel arrived at destination port', 'Discharged from vessel at destination port', 'Gate out from destination port', 'Gate in empty return']


# In[6]:


c = df[df.EVENT_DESCRIPTION.isin(core)]


# In[7]:


c.head(5)


# In[8]:


c.shape


# In[9]:


c.drop_duplicates(inplace=True)


# In[10]:


c.shape


# .

# ### Task 1:
# 
# There are eight core events expected to occur for each journey identified by “reference_id”. What are the rates for how often these events are present for each shipment? You may need to consider whether the container journey is completed for more recent shipments.
# 
# - overall
# - overall per ocean carrier
# - per each of the 8 core events overall
# - per each of the 8 core events per ocean carrier

# In[11]:


# Number of core events completed by shipment

c['CORE_EVENTS_COMPLETED'] = c.groupby('REFERENCE_ID')['EVENT_DESCRIPTION'].transform('nunique')


# In[12]:


# Overall rate for how often all eight of these core events are present for a shipment

print('Overall rate for 8 core events to be present for a shipment:',round(c[c.CORE_EVENTS_COMPLETED == 8]['REFERENCE_ID'].nunique() / c['REFERENCE_ID'].nunique() * 100,2),'%')


# In[13]:


# Overall rate for how often all eight of the core events are present for each shipment per carrier

pd.DataFrame(round(c[c.CORE_EVENTS_COMPLETED == 8].groupby('CARRIER_NAME')['REFERENCE_ID'].nunique() / c.groupby('CARRIER_NAME')['REFERENCE_ID'].nunique()* 100,2)
             .sort_values(ascending=False)).reset_index().rename(columns={'CARRIER_NAME':'CARRIER','REFERENCE_ID':'% RATE OF COMPLETED JOURNEY'}).fillna(0.00)


# In[14]:


# Overall rate for how often these events are present for each shipment

pd.DataFrame(round(c[c.CORE_EVENTS_COMPLETED == 8].groupby('EVENT_DESCRIPTION')['CORE_EVENTS_COMPLETED'].count() / c.groupby('EVENT_DESCRIPTION')['CORE_EVENTS_COMPLETED'].count() * 100,2)
             .sort_values(ascending=False)).reset_index().rename(columns= {'EVENT_DESCRIPTION':'EVENT TYPE','CORE_EVENTS_COMPLETED':'% RATE OF COMPLETED EVENT'})


# In[15]:


pd.DataFrame(round(c[c.CORE_EVENTS_COMPLETED == 8].groupby(['CARRIER_NAME','EVENT_DESCRIPTION'])['CORE_EVENTS_COMPLETED'].nunique() / c.groupby(['CARRIER_NAME','EVENT_DESCRIPTION'])['CORE_EVENTS_COMPLETED'].nunique() * 100,2)).reset_index().sort_values(by=['CARRIER_NAME','CORE_EVENTS_COMPLETED'],ascending=[True,False]).rename(columns= {'CORE_EVENTS_COMPLETED':'PERCENT_PRESENT'})


# .

# ### Task 2:
# 
# Import dwell time is calculated as the time from [Discharged from vessel at destination port] to [Gate out from destination port], giving you the time a container dwelled in a terminal before an out gate to the consignee. For shipments that have both of these events can you determine the import dwell time per shipment?EVENT_DESCRIPTION

# In[16]:


def dwell_time(df):
    
    
    #ths function calulates and outputs the time acontainer dwelled in a terminal before an out gate to the consignee
    
    # isolate columns for dataframe
    dt = df.copy()
    
    dt = dt[['REFERENCE_ID','CONTAINER_ID','EVENT_DESCRIPTION','EVENT_TIMESTAMP']]
    
    # change event timestamp from object to datetime
    # change datetime format from UTC
    dt['EVENT_DATETIME'] = pd.to_datetime(dt['EVENT_TIMESTAMP'])
    dt['EVENT_DATETIME'] = pd.to_datetime(dt.EVENT_DATETIME).dt.tz_localize(None)
    
    # isolate events and drop duplicated rows
    dt = dt[(dt['EVENT_DESCRIPTION'] == 'Discharged from vessel at destination port') | (dt['EVENT_DESCRIPTION']== 'Gate out from destination port')]
    dt.drop_duplicates(inplace=True)
    
    # group the shipments and isolate which have equal to or less than two unique timestamps
    # this allows us to have the two events per shipment,if applicable 
    timestamps = dt.groupby('REFERENCE_ID').filter(lambda x: len(np.unique(x['EVENT_DATETIME']))<=2).REFERENCE_ID

    # create a new dataframe with the just the shipments with two unique timestamps
    # add timestasmps to new column by grouping the reference ids
    dt2 = dt[dt['REFERENCE_ID'].isin(timestamps)].copy().sort_values(by=['REFERENCE_ID','EVENT_DATETIME'])
    
    for name, group in dt2.groupby('REFERENCE_ID'):
        for i in range(1, len(group)):
            dt2.loc[group.index[0], 'EVENT_DATETIME'+str(i+1)] = group.EVENT_DATETIME.iloc[i]
      
    # rename columns
    dt2.rename(columns={'EVENT_DATETIME2':'OUT','EVENT_DATETIME':'DISCHARGED'},inplace=True)
     
    # caluclate dwell time from when the shipment was discharged to an out gate
    dt2['DWELL_TIME'] = dt2['OUT'] - dt2['DISCHARGED'] 
    
    #drop na to keep one complete row per shipment
    dt2.dropna(how='any',inplace= True)
    
    # dataframe output
    final = pd.DataFrame(dt2.groupby(['REFERENCE_ID','CONTAINER_ID'])['DWELL_TIME'].sum().sort_values(ascending=False)).reset_index()
    
    return final


# In[17]:


dw = dwell_time(c)


# In[18]:


dw


# .

# ### Task 3:
# 
# Are origin and destination port values changing throughout the journey of a container? If so, how many container journeys are experiencing this?

# In[26]:


def port_values(x):
    vc = pd.DataFrame(x.groupby('REFERENCE_ID')[['DESTINATION_PORT','ORIGIN_PORT']].nunique()).reset_index()
    print('.......................................................')
    print(vc[(vc['DESTINATION_PORT'] != 1) & (vc['ORIGIN_PORT'] == 1)]['REFERENCE_ID'].count(), 'container journeys had a changed Destination Port value.')
    print(vc[(vc['DESTINATION_PORT'] == 1) & (vc['ORIGIN_PORT'] != 1)]['REFERENCE_ID'].count(),'container journeys had a changed Origin Port value.')
    print(vc[(vc['DESTINATION_PORT'] != 1) & (vc['ORIGIN_PORT'] != 1)]['REFERENCE_ID'].count(),'container journeys had changed Destination Port and Origin Port values.')
    print(vc[(vc['DESTINATION_PORT'] == 1) & (vc['ORIGIN_PORT'] == 1)]['REFERENCE_ID'].count(), 'container journeys had consistent Destination Port and Origin Port values.')
    print('.......................................................')
    return vc


# In[27]:


port_values(c)


# .

# ### Task 4:
# 
# Create a matrix that shows all events reported (not just the 8 core ocean). For these events show:
# 
# - which carriers report the event
# - if the event has been reported as an estimate, an actual, or both

# In[28]:


matrix = df.copy()


# In[29]:


matrix['PLANNED_NUM'] = np.where(matrix['PLANNED'] == True,1, 0)


# In[30]:


output = matrix.pivot_table(index=['CARRIER_NAME','EVENT_DESCRIPTION'], columns=['PLANNED'],values='PLANNED_NUM',aggfunc="count",fill_value=0)


# In[31]:


output


# .

# In[54]:


#Bonus - - Calculate import dwell time per destination port to determine which ports may perform better in regard to container throughput.
# I extracted (what I think is) the port code, though I would need to work on the spacing & consistency
# Needs further time, just an idea that is being flushed out

import re


# In[33]:


df['PORT_CODE'] = df['DESTINATION_PORT'].str.replace('\W', ' ')


# In[34]:


df['PORT_CODE'].fillna('None',inplace=True)


# In[35]:


filtered_df = df[df['PORT_CODE'].str.contains('unlocode', flags=re.IGNORECASE)]


# In[36]:


df.loc[filtered_df.index, 'PORT_CODE'] = [re.split('unlocode', row, flags=re.IGNORECASE, maxsplit=1)[-1] for row in filtered_df['PORT_CODE']]


# In[37]:


df.head()


# In[38]:


k = df[['REFERENCE_ID','CONTAINER_ID','PORT_CODE']]


# In[39]:


k = k.drop_duplicates()


# In[40]:


k


# In[41]:


dw


# In[42]:


test = pd.merge(k, dw, on=['REFERENCE_ID','CONTAINER_ID'], how="left")


# In[43]:


test


# In[44]:


test.dropna(how='any',inplace=True)


# In[45]:


test = test.drop_duplicates()


# In[46]:


test


# In[53]:


print('The most efficient ports with the shortest total dwell time per shipment\:',test.groupby(['CONTAINER_ID','PORT_CODE'])['DWELL_TIME'].sum().sort_values().nsmallest(5))


# In[ ]:




