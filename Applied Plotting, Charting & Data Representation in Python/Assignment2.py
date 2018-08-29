
# coding: utf-8

# # Assignment 2
# 
# Before working on this assignment please read these instructions fully. In the submission area, you will notice that you can click the link to **Preview the Grading** for each step of the assignment. This is the criteria that will be used for peer grading. Please familiarize yourself with the criteria before beginning the assignment.
# 
# An NOAA dataset has been stored in the file `data/C2A2_data/BinnedCsvs_d400/fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv`. The data for this assignment comes from a subset of The National Centers for Environmental Information (NCEI) [Daily Global Historical Climatology Network](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt) (GHCN-Daily). The GHCN-Daily is comprised of daily climate records from thousands of land surface stations across the globe.
# 
# Each row in the assignment datafile corresponds to a single observation.
# 
# The following variables are provided to you:
# 
# * **id** : station identification code
# * **date** : date in YYYY-MM-DD format (e.g. 2012-01-24 = January 24, 2012)
# * **element** : indicator of element type
#     * TMAX : Maximum temperature (tenths of degrees C)
#     * TMIN : Minimum temperature (tenths of degrees C)
# * **value** : data value for element (tenths of degrees C)
# 
# For this assignment, you must:
# 
# 1. Read the documentation and familiarize yourself with the dataset, then write some python code which returns a line graph of the record high and record low temperatures by day of the year over the period 2005-2014. The area between the record high and record low temperatures for each day should be shaded.
# 2. Overlay a scatter of the 2015 data for any points (highs and lows) for which the ten year record (2005-2014) record high or record low was broken in 2015.
# 3. Watch out for leap days (i.e. February 29th), it is reasonable to remove these points from the dataset for the purpose of this visualization.
# 4. Make the visual nice! Leverage principles from the first module in this course when developing your solution. Consider issues such as legends, labels, and chart junk.
# 
# The data you have been given is near **Ann Arbor, Michigan, United States**, and the stations the data comes from are shown on the map below.

# In[4]:

import matplotlib.pyplot as plt
import mplleaflet
import pandas as pd

def leaflet_plot_stations(binsize, hashid):

    df = pd.read_csv('data/C2A2_data/BinSize_d{}.csv'.format(binsize))

    station_locations_by_hash = df[df['hash'] == hashid]

    lons = station_locations_by_hash['LONGITUDE'].tolist()
    lats = station_locations_by_hash['LATITUDE'].tolist()

    plt.figure(figsize=(8,8))

    plt.scatter(lons, lats, c='r', alpha=0.7, s=200)

    return mplleaflet.display()

leaflet_plot_stations(400,'fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89')


# In[87]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import mplleaflet
import pandas as pd
import numpy as np

def max_min_temp_scatter_plot(binsize, hashid):
    df = pd.read_csv('data/C2A2_data/BinnedCsvs_d{}/{}.csv'.format(binsize,hashid))
    
    #2004-2015 Data without leap days 
    df['Date'] = pd.to_datetime(df['Date'])
    rel_data = df[(df['Date'].dt.year>=2005) & (df['Date'].dt.year<2015)]
    rel_data = rel_data[~(df['Date'].dt.is_leap_year)]
    #2015 Data
    max_df_2015 = df[(df['Date'].dt.year==2015) & (df['Element']=='TMAX')]
    min_df_2015 = df[(df['Date'].dt.year==2015) & (df['Element']=='TMIN')]
    
    fig, ax = plt.subplots(figsize=(15,8))
    max_temp = rel_data.groupby([rel_data['Date'].dt.month,rel_data['Date'].dt.day])['Data_Value'].agg(['max'])
    min_temp = rel_data.groupby([rel_data['Date'].dt.month,rel_data['Date'].dt.day])['Data_Value'].agg(['min'])
    max_grouped_df = max_df_2015.groupby([max_df_2015['Date'].dt.month,max_df_2015['Date'].dt.day])['Data_Value'].agg(['max'])
    min_grouped_df = min_df_2015.groupby([min_df_2015['Date'].dt.month,min_df_2015['Date'].dt.day])['Data_Value'].agg(['min'])
    
    #Changing Indexes
    max_temp.index.names = ['Month', 'Day']
    min_temp.index.names = ['Month', 'Day']
    max_grouped_df.index.names = ['Month', 'Day']
    min_grouped_df.index.names = ['Month', 'Day']
    
    
    max_temp.reset_index(inplace= True)
    min_temp.reset_index(inplace= True)
    max_grouped_df.reset_index(inplace= True)
    min_grouped_df.reset_index(inplace= True)
    

    #Index Formation
    max_temp["Month,Day"] = max_temp['Month'].map(str)+", " + max_temp['Day'].map(str)
    min_temp["Month,Day"] = min_temp['Month'].map(str)+", " + min_temp['Day'].map(str)
    
    max_grouped_df["Month,Day"] = max_grouped_df['Month'].map(str)+", " + max_grouped_df['Day'].map(str)
    min_grouped_df["Month,Day"] = min_grouped_df['Month'].map(str)+", " + min_grouped_df['Day'].map(str)
    
    # Extreme Points
    max_grouped_df["flag"] = max_grouped_df['max']>max_temp["max"]
    min_grouped_df["flag"] = min_grouped_df['min']<min_temp["min"]
    
    max_grouped_df = max_grouped_df[max_grouped_df['flag']==True]
    min_grouped_df = min_grouped_df[min_grouped_df['flag']==True]
    
    
    #Plotting Code
    plt.plot(max_temp.index, max_temp["max"],  label = "Maximum Temperature (Over the period of 2005-2014)", alpha = 0.7)
    plt.plot(min_temp.index, min_temp["min"],  label = "Minimum Temperature (Over the period of 2005-2014)", alpha = 0.7)
    plt.fill_between(min_temp.index, min_temp["min"],max_temp["max"], color= 'lightgrey', alpha = 0.4 )
    plt.scatter(max_grouped_df.index,max_grouped_df['max'] ,s=30, color = 'r',label = "Extreme Temperature in Year 2015")
    plt.scatter(min_grouped_df.index,min_grouped_df['min'] ,s=30, color = 'b',label = "Extreme Temperature in Year 2015")
    
    # Title and Labeling Code
    plt.title("Extreme Temperatures in 'Ann Arbor, Michigan, United States' region in Year 2015")
    plt.ylabel("Temperature (tenths of degree C)")
    plt.xlabel("Month, Day")
    month_dict = {"1":"Jan, 01", "2":"Feb, 01", "3":"Mar, 01", "4":"Apr, 01","5":"May, 01","6":"Jun, 01","7":"Jul, 01", "8":"Aug, 01","9":"Sep, 01","10":"Oct, 01", "11":"Nov, 01","12":"Dec, 01"}
    xtiks = set()
    labels_list = []
    ticks_list = []
    for i in range(0,365):
        idx = min_temp["Month,Day"][i].split(",")[0]
        #print(idx)
        if idx not in xtiks:
            #print(min_temp["Month,Day"][i].split(",")[0])
            ticks_list.append(i)
            xtiks.add(idx)
            labels_list.append(month_dict[idx])
            
    ax.set_xticks(ticks_list)
    ax.set_xticklabels(labels_list, rotation=0)
    plt.margins(x=0)
    plt.legend(frameon=False)
    plt.tick_params(top="off", bottom = "on", left="on")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()
    
    return

max_min_temp_scatter_plot(400,'fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89')


# In[ ]:




# In[ ]:



