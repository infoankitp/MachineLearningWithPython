
# coding: utf-8

# # Assignment 4
# 
# Before working on this assignment please read these instructions fully. In the submission area, you will notice that you can click the link to **Preview the Grading** for each step of the assignment. This is the criteria that will be used for peer grading. Please familiarize yourself with the criteria before beginning the assignment.
# 
# This assignment requires that you to find **at least** two datasets on the web which are related, and that you visualize these datasets to answer a question with the broad topic of **economic activity or measures** (see below) for the region of **Ann Arbor, Michigan, United States**, or **United States** more broadly.
# 
# You can merge these datasets with data from different regions if you like! For instance, you might want to compare **Ann Arbor, Michigan, United States** to Ann Arbor, USA. In that case at least one source file must be about **Ann Arbor, Michigan, United States**.
# 
# You are welcome to choose datasets at your discretion, but keep in mind **they will be shared with your peers**, so choose appropriate datasets. Sensitive, confidential, illicit, and proprietary materials are not good choices for datasets for this assignment. You are welcome to upload datasets of your own as well, and link to them using a third party repository such as github, bitbucket, pastebin, etc. Please be aware of the Coursera terms of service with respect to intellectual property.
# 
# Also, you are welcome to preserve data in its original language, but for the purposes of grading you should provide english translations. You are welcome to provide multiple visuals in different languages if you would like!
# 
# As this assignment is for the whole course, you must incorporate principles discussed in the first week, such as having as high data-ink ratio (Tufte) and aligning with Cairo’s principles of truth, beauty, function, and insight.
# 
# Here are the assignment instructions:
# 
#  * State the region and the domain category that your data sets are about (e.g., **Ann Arbor, Michigan, United States** and **economic activity or measures**).
#  * You must state a question about the domain category and region that you identified as being interesting.
#  * You must provide at least two links to available datasets. These could be links to files such as CSV or Excel files, or links to websites which might have data in tabular form, such as Wikipedia pages.
#  * You must upload an image which addresses the research question you stated. In addition to addressing the question, this visual should follow Cairo's principles of truthfulness, functionality, beauty, and insightfulness.
#  * You must contribute a short (1-2 paragraph) written justification of how your visualization addresses your stated research question.
# 
# What do we mean by **economic activity or measures**?  For this category you might look at the inputs or outputs to the given economy, or major changes in the economy compared to other regions.
# 
# ## Tips
# * Wikipedia is an excellent source of data, and I strongly encourage you to explore it for new data sources.
# * Many governments run open data initiatives at the city, region, and country levels, and these are wonderful resources for localized data sources.
# * Several international agencies, such as the [United Nations](http://data.un.org/), the [World Bank](http://data.worldbank.org/), the [Global Open Data Index](http://index.okfn.org/place/) are other great places to look for data.
# * This assignment requires you to convert and clean datafiles. Check out the discussion forums for tips on how to do this from various sources, and share your successes with your fellow students!
# 
# ## Example
# Looking for an example? Here's what our course assistant put together for the **Ann Arbor, MI, USA** area using **sports and athletics** as the topic. [Example Solution File](./readonly/Assignment4_example.pdf)

# In[78]:

import pandas as pd


# In[79]:

household_income_real = [[441784800000, 50546],
 [473407200000, 51586],
 [504943200000, 55640],
 [536479200000, 56008],
 [568015200000, 57509],
 [599637600000, 57536],
 [631173600000, 53339],
 [662709600000, 55214],
 [694245600000, 54126],
 [725868000000, 53441],
 [757404000000, 56551],
 [788940000000, 57008],
 [820476000000, 59796],
 [852098400000, 57810],
 [883634400000, 61570],
 [915170400000, 66439],
 [946706400000, 63454],
 [978328800000, 61067],
 [1009864800000, 57007],
 [1041400800000, 58752],
 [1072936800000, 53692],
 [1104559200000, 56452],
 [1136095200000, 57910],
 [1167631200000, 57150],
 [1199167200000, 55502],
 [1230789600000, 51451],
 [1262325600000, 50943],
 [1293861600000, 52147],
 [1325397600000, 52284],
 [1357020000000, 58286],
 [1388556000000, 52723],
 [1420092000000, 54888],
 [1451628000000, 57091]]
household_income_real_df = pd.DataFrame(household_income_real)
household_income_real_df.columns = ['Year', 'Ann Arbor Real Income']
household_income_real_df['Year'] = pd.to_datetime(household_income_real_df['Year'], unit= 'ms').dt.year
household_income_real_df.set_index('Year', inplace = True)
#household_income_real_df.head()

household_income_real_df = household_income_real_df.loc[2005:]
household_income_real_df.head()
#household_income_real_df


# In[80]:

household_income_adjusted = [[2016,57617,52492,65601],
[2015,56480,51729,62760],
[2014,54398,50535,63713],
[2013,53838,49741,61474],
[2012,53701,48984,58885],
[2011,53879,49056,60398],
[2010,55093,49992,61515],
[2009,56180,50624,61082],
[2008,58000,54168,64487],
[2007,58736,55506,70669],
[2006,57677,56166,67636],
[2005,56832,56582,65746]]
household_income_adjusted_df = pd.DataFrame(household_income_adjusted)
household_income_adjusted_df.columns = ["Year", "US Inflation Adjusted Income", "Michigan Inflation Adjusted Income", "Ann Arbor Inflation Adjusted Income"]
household_income_adjusted_df.set_index("Year", inplace = True)


# In[81]:

get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt


# In[88]:

ax = household_income_adjusted_df.plot()
household_income_real_df.plot(ax = ax)
ax.set_title("Household Income Since 2005 Inflation Adjusted and Real");
#ax.fill_between(household_income_adjusted_df.index,household_income_real_df['Ann Arbor Real Income'], household_income_adjusted_df['Ann Arbor Inflation Adjusted Income'],alpha = 0.7, color = 'lightslategray', 
#);
ax.set_ylabel("Median Income in US$")
ax.legend(frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax.tick_params(left='on')


# In[ ]:



