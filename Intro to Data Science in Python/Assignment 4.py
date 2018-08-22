
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# In[2]:

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import re


# # Assignment 4 - Hypothesis Testing
# This assignment requires more individual learning than previous assignments - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.
# 
# Definitions:
# * A _quarter_ is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.
# * A _recession_ is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
# * A _recession bottom_ is the quarter within a recession which had the lowest GDP.
# * A _university town_ is a city which has a high percentage of university students compared to the total population of the city.
# 
# **Hypothesis**: University towns have their mean housing prices less effected by recessions. Run a t-test to compare the ratio of the mean price of houses in university towns the quarter before the recession starts compared to the recession bottom. (`price_ratio=quarter_before_recession/recession_bottom`)
# 
# The following data files are available for this assignment:
# * From the [Zillow research data site](http://www.zillow.com/research/data/) there is housing data for the United States. In particular the datafile for [all homes at a city level](http://files.zillowstatic.com/research/public/City/City_Zhvi_AllHomes.csv), ```City_Zhvi_AllHomes.csv```, has median home sale prices at a fine grained level.
# * From the Wikipedia page on college towns is a list of [university towns in the United States](https://en.wikipedia.org/wiki/List_of_college_towns#College_towns_in_the_United_States) which has been copy and pasted into the file ```university_towns.txt```.
# * From Bureau of Economic Analysis, US Department of Commerce, the [GDP over time](http://www.bea.gov/national/index.htm#gdp) of the United States in current dollars (use the chained value in 2009 dollars), in quarterly intervals, in the file ```gdplev.xls```. For this assignment, only look at GDP data from the first quarter of 2000 onward.
# 
# Each function in this assignment below is worth 10%, with the exception of ```run_ttest()```, which is worth 50%.

# In[51]:

# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}


# In[106]:

def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the 
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
    columns=["State", "RegionName"]  )
    
    The following cleaning needs to be done:

    1. For "State", removing characters from "[" to the end.
    2. For "RegionName", when applicable, removing every character from " (" to the end.
    3. Depending on how you read the data, you may need to remove newline character '\n'. '''
    text = None
    with open("university_towns.txt", "r") as f:
        text = f.read()
    text = text[:-2]
    lines = text.split("\n")
    count =0
    data_list = []
    for line in lines:
        if '[edit]' in line: 
            state = line[:-6]
        elif '(' in line:
            region_name = line.split(" (")[0]
            if region_name is not None and region_name != "":
                data_list.append([state,region_name])
        else:
            data_list.append([state,line])
    rslt = pd.DataFrame(data_list,columns = ['State','RegionName'])
    return rslt
#get_list_of_university_towns()


# In[23]:

df = pd.read_excel("gdplev.xls",skiprows = 5)
df1 = df[["Unnamed: 4", "GDP in billions of current dollars.1"]]
df1.columns = ["Quarter","GDP"]
df = df1.dropna()
df['GDP'] = pd.to_numeric(df['GDP'])


# In[25]:

def get_recession_start():
    '''Returns the year and quarter of the recession start time as a 
    string value in a format such as 2005q3'''
    # Assuming We need to find out the Biggest Reccession Start Quarter
    rslt = None
    max_diff = 0
    for i in range(0,len(df)-2):
        
        if (df.iloc[i+1]["GDP"]  < df.iloc[i]["GDP"] ) and (df.iloc[i+2]["GDP"] < df.iloc[i+1]["GDP"] ):
            diff = (df.iloc[i+1]["GDP"] - df.iloc[i+2]["GDP"]) + (df.iloc[i]["GDP"]-df.iloc[i+1]["GDP"])
            if rslt != df.iloc[i-1]["Quarter"] and diff>max_diff :
                rslt = df.iloc[i]["Quarter"]
                max_diff = diff
           
    return rslt

#get_recession_start()


# In[ ]:




# In[26]:

def get_recession_end():
    '''Returns the year and quarter of the recession end time as a 
    string value in a format such as 2005q3'''
    start = get_recession_start()
    start_index = df[df["Quarter"] == start].index.values[0]
    rslt = None
    for i in range(start_index+1, len(df)-2):
        if (df.iloc[i+1]["GDP"]  > df.iloc[i]["GDP"] ) and (df.iloc[i+2]["GDP"] > df.iloc[i+1]["GDP"] ):
            rslt = df.iloc[i+2]["Quarter"]
            break
    return rslt
#get_recession_end()


# In[28]:

def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a 
    string value in a format such as 2005q3'''
    start_index = df[df["Quarter"] == get_recession_start()].index.values[0]
    end_index = df[df["Quarter"] == get_recession_end()].index.values[0]
    min_gdp = 9999999999999999
    for i in range(start_index,end_index):
        if df.iloc[i]["GDP"]< min_gdp:
            rslt = df.iloc[i]["Quarter"]
            min_gdp = df.iloc[i]["GDP"]
    return rslt
#get_recession_bottom()


# In[30]:

housing_df = pd.read_csv("City_Zhvi_AllHomes.csv")
#housing_df.head()


# In[131]:

def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.
    
    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''
    rslt = housing_df[["State","RegionName"]]
    
    df1 = housing_df.set_index(["State","RegionName"])
    rslt = rslt.set_index(["State","RegionName"])
    for year in range(2000,2017):
        for quarter in range(1,5):
            if year == 2016 and quarter ==3:
                rslt[str(year)+"q"+str(quarter)] = (df1[str(year)+"-%02d" % int(1+ (quarter-1)*3)]+df1[str(year)+"-%02d" % int(2+ (quarter-1)*3)])/2
                break
            else:
                rslt[str(year)+"q"+str(quarter)] = (df1[str(year)+"-%02d" % int(1+ (quarter-1)*3)]+df1[str(year)+"-%02d" % int(2+ (quarter-1)*3)]+df1[str(year)+"-%02d" % int(3+ (quarter-1)*3)])/3
    rslt.reset_index(inplace =True)
    rslt.replace({"State":states},inplace = True)
    rslt.set_index(["State","RegionName"], inplace= True)
    return rslt
#convert_housing_data_to_quarters()


# In[132]:

from scipy import stats
def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''
    start_year_quarter = get_recession_start()
    starting_year = int(start_year_quarter[:4])
    starting_quarter = int(start_year_quarter[-1])
    bottom_year_quarter = get_recession_bottom()
    bottom_year = int(bottom_year_quarter[:4])
    bottom_quarter = int(bottom_year_quarter[-1])
    housing_data = convert_housing_data_to_quarters()
    rel_data = pd.DataFrame(index = housing_data.index)
    #rel_data.set_index(["State","RegionName"],inplace=True)
#     print(rel_data[["Wisconsin","Holland"]])
    for year in range(starting_year,bottom_year+1):
        for quarter in range(1,5):
            if (year == starting_year and quarter < starting_quarter) or (year == bottom_year and quarter > bottom_quarter):
                continue
            else:
                #print(str(year)+"q"+str(quarter))
                rel_data[str(year)+"q"+str(quarter)] = housing_data[str(year)+"q"+str(quarter)]
        pass
    def price_ratio(row,start_quarter, bottom_quarter):
        return (row[start_quarter] - row[bottom_quarter])/row[start_quarter]
    
    rel_data['price_ratio'] = rel_data.apply(price_ratio,args=(start_year_quarter,bottom_year_quarter),axis=1)
    data = rel_data.reset_index()
    univ_town_list = get_list_of_university_towns().copy()['RegionName']
    univ_set = set(univ_town_list)
    def is_univ_town(row):
        #check if the town is a university towns or not.
        if row['RegionName'] in univ_set:
            return 1
        else:
            return 0
    data['is_univ_town'] = data.apply(is_univ_town,axis=1)
    
    univ_town = data[data['is_univ_town']==1].loc[:,'price_ratio'].dropna()
    not_univ_town = data[data['is_univ_town']!=1].loc[:,'price_ratio'].dropna()
    def is_better():
        if not_univ_town.mean() > univ_town.mean():
            return 'university town'
        else:
            return 'non-university town'
        
    test_value = stats.ttest_ind(not_univ_town,univ_town)
    p_value = test_value[1]
    if p_value < 0.1:
        different =True
    else:
        different = False
    
    return (different, p_value, is_better())
#run_ttest()


# In[ ]:



