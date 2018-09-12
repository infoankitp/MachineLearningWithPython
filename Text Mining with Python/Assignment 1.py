
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 1
# 
# In this assignment, you'll be working with messy medical data and using regex to extract relevant infromation from the data. 
# 
# Each line of the `dates.txt` file corresponds to a medical note. Each note has a date that needs to be extracted, but each date is encoded in one of many formats.
# 
# The goal of this assignment is to correctly identify all of the different date variants encoded in this dataset and to properly normalize and sort the dates. 
# 
# Here is a list of some of the variants you might encounter in this dataset:
# * 04/20/2009; 04/20/09; 4/20/09; 4/3/09
# * Mar-20-2009; Mar 20, 2009; March 20, 2009;  Mar. 20, 2009; Mar 20 2009;
# * 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
# * Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
# * Feb 2009; Sep 2009; Oct 2010
# * 6/2008; 12/2009
# * 2009; 2010
# 
# Once you have extracted these date patterns from the text, the next step is to sort them in ascending chronological order accoring to the following rules:
# * Assume all dates in xx/xx/xx format are mm/dd/yy
# * Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
# * If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
# * If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).
# * Watch out for potential typos as this is a raw, real-life derived dataset.
# 
# With these rules in mind, find the correct date in each note and return a pandas Series in chronological order of the original Series' indices.
# 
# For example if the original series was this:
# 
#     0    1999
#     1    2010
#     2    1978
#     3    2015
#     4    1985
# 
# Your function should return this:
# 
#     0    2
#     1    4
#     2    0
#     3    1
#     4    3
# 
# Your score will be calculated using [Kendall's tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient), a correlation measure for ordinal data.
# 
# *This function should return a Series of length 500 and dtype int.*

# In[86]:

import pandas as pd
import numpy as np
doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)
df


# In[94]:

def date_sorter():
    month_dict = {'Jan' : 1 , 'Feb' : 2, 'Mar' : 3, 'Apr' : 4, 'May' : 5,'Jun':6, 'Jul' : 7,'Aug' : 8, 'Sep' : 9, 'Oct' : 10, 'Nov' : 11, 'Dec' : 12, 'Age' : 1}
    ser = df.str.strip()

    rslt = ser.str.extractall(r'(?P<Origin>(?P<Month>\d?\d)[\/\-](?P<Day>\d?\d)[\/\-](?P<Year>\d{4})){1}')
    index_left = ~df.index.isin([x[0] for x in rslt.index.tolist()])
    df_left = df[index_left]
#     rslt = rslt.append(df[index_left].str.extractall(r'(?P<origin>(?P<month>\d?\d)[/|-](?P<day>([0-2]?[0-9])|([3][01]))[/|-](?P<year>\d{2}))'))
    rslt = rslt.append(df_left.str.extractall(r"(?P<Origin>(?P<Month>\d?\d)[\/\-](?P<Day>\d?\d)[\/\-](?P<Year>\d{2})){1}"))
    index_left = ~df.index.isin([x[0] for x in rslt.index.tolist()])
    df_left = df[index_left]
    
    rslt = rslt.append(df_left.str.extractall(r"(?P<Origin>(?P<Month>[A-Z][a-z]{2})[- ](?P<Day>[0-3]?\d)[- ](?P<Year>\d{4}))"))
    index_left = ~df.index.isin([x[0] for x in rslt.index.tolist()])
    df_left = df[index_left]
    
    rslt = rslt.append(df_left.str.extractall(r"(?P<Origin>(?P<Month>[A-Z][a-z]{2,})[- ](?P<Day>[0-3]?\d)[- ](?P<Year>\d{4}))"))
    index_left = ~df.index.isin([x[0] for x in rslt.index.tolist()])
    df_left = df[index_left]
    
    rslt = rslt.append(df_left.str.extractall(r"(?P<Origin>(?P<Day>[0-3]?\d),?.? (?P<Month>[A-Z][a-z]{2,}),?.? (?P<Year>\d{4}))"))
    index_left = ~df.index.isin([x[0] for x in rslt.index.tolist()])
    df_left = df[index_left]
    
    rslt = rslt.append(df_left.str.extractall(r"(?P<Origin>(?P<Month>[A-Z][a-z]{2,}),?.? (?P<Day>[0-3]?\d),?.? (?P<Year>\d{4}))"))
    index_left = ~df.index.isin([x[0] for x in rslt.index.tolist()])
    df_left = df[index_left]
    
    rslt = rslt.append(df_left.str.extractall(r"(?P<Origin>(?P<Day>[0-3]?\d),?.? (?P<Month>[A-Z][a-z]{,2}),?.? (?P<Year>\d{4}))"))
    index_left = ~df.index.isin([x[0] for x in rslt.index.tolist()])
    df_left = df[index_left]
    
    rslt = rslt.append(df_left.str.extractall(r"(?P<Origin>(?P<Month>[A-Z][a-z]{,2}),?.? (?P<Day>[0-3]?\d),?.? (?P<Year>\d{4}))"))
    index_left = ~df.index.isin([x[0] for x in rslt.index.tolist()])
    df_left = df[index_left]
#     Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
    
    rslt = rslt.append(df_left.str.extractall(r"(?P<Origin>(?P<Month>[A-Z][a-z]{2}),?.? (?P<Day>[0-3]?\d)[a-z]{,2},?.? (?P<Year>\d{4}))"))
    index_left = ~df.index.isin([x[0] for x in rslt.index.tolist()])
    df_left = df[index_left]
        
#     Feb 2009; Sep 2009; Oct 2010
    
    rslt = rslt.append(df_left.str.extractall(r"(?P<Origin>(?P<Month>[A-Z][a-z]{2,}),?.? (?P<Year>\d{4}))"))
    index_left = ~df.index.isin([x[0] for x in rslt.index.tolist()])
    df_left = df[index_left]
    
    rslt = rslt.append(df_left.str.extractall(r"(?P<Origin>(?P<Month>[0-1]?\d/?)/(?P<Year>\d{4}))"))
    index_left = ~df.index.isin([x[0] for x in rslt.index.tolist()])
    df_left = df[index_left]
    
    rslt = rslt.append(df_left.str.extractall(r"(?P<Origin>(?P<Month>[0-1]?\d)(?P<Year>\d{4}))"))
    index_left = ~df.index.isin([x[0] for x in rslt.index.tolist()])
    df_left = df[index_left]
    
    rslt = rslt.append(df_left.str.extractall(r"(?P<Year>\d{4})"))
    index_left = ~df.index.isin([x[0] for x in rslt.index.tolist()])
    df_left = df[index_left]
    rslt['Month'] = rslt["Month"].fillna("01")
    rslt['Month'] = rslt['Month'].str.lstrip('s')
    rslt['Month'] = rslt['Month'].str.lstrip('l')
    rslt['Month'] = rslt['Month'].str.lstrip('y')
    rslt['Month'] = rslt['Month'].str.lstrip('s')
    rslt["New_Month"] = rslt["Month"].apply(lambda x: x if x.isdigit() else ( month_dict[x[0:3]] if x[0:3] in month_dict.keys() else 1 )  )
    rslt["Day"] = rslt["Day"].fillna("01")
    rslt["Year"] = rslt["Year"].apply(lambda x: x if len(str(int(x)))!=2 else "19"+str(int(x)) )
    rslt["Day"] = rslt["Day"].apply(lambda x: x if int(x) <32 else np.NaN)
    
    rslt["New_Month"] = rslt["New_Month"].apply(lambda x: '{:02d}'.format(int(x)))
    rslt = rslt.dropna(axis = 0, subset = ["Day"])
    rslt["Date"] = pd.to_datetime((rslt.Year.astype(str)+"-"+rslt.New_Month.astype(str)+"-"+rslt.Day.astype(str)).apply(str),format = "%Y-%m-%d")
    
    rslt.sort_values(by='Date', inplace=True)
    
    df1 = pd.Series(list(rslt.index.labels[0]))
    return df1
date_sorter()


# In[ ]:




# In[ ]:



