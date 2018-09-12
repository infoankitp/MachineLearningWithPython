
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 2 - Introduction to NLTK
# 
# In part 1 of this assignment you will use nltk to explore the Herman Melville novel Moby Dick. Then in part 2 you will create a spelling recommender function that uses nltk to find words similar to the misspelling. 

# ## Part 1 - Analyzing Moby Dick

# In[1]:

import nltk
import pandas as pd
import numpy as np

# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)


# ### Example 1
# 
# How many tokens (words and punctuation symbols) are in text1?
# 
# *This function should return an integer.*

# In[2]:

def example_one():
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)

example_one()


# ### Example 2
# 
# How many unique tokens (unique words and punctuation) does text1 have?
# 
# *This function should return an integer.*

# In[3]:

def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))

example_two()


# ### Example 3
# 
# After lemmatizing the verbs, how many unique tokens does text1 have?
# 
# *This function should return an integer.*

# In[ ]:

from nltk.stem import WordNetLemmatizer

def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))

example_three()


# ### Question 1
# 
# What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)
# 
# *This function should return a float.*

# In[4]:

def answer_one():
    
    
    return example_two()/example_one()

answer_one()


# ### Question 2
# 
# What percentage of tokens is 'whale'or 'Whale'?
# 
# *This function should return a float.*

# In[10]:

def answer_two():
    
    dist = nltk.FreqDist(text1)
    return (dist["whale"]+dist["Whale"])*100/len(text1)

answer_two()


# ### Question 3
# 
# What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
# 
# *This function should return a list of 20 tuples where each tuple is of the form `(token, frequency)`. The list should be sorted in descending order of frequency.*

# In[28]:

def answer_three():
    dist = nltk.FreqDist(text1)
    sorted_dist = [(k, dist[k]) for k in sorted(dist, reverse = True, key = dist.get )]
    return sorted_dist[:20]#sorted_dist[:20]

answer_three()


# ### Question 4
# 
# What tokens have a length of greater than 5 and frequency of more than 150?
# 
# *This function should return a sorted list of the tokens that match the above constraints. To sort your list, use `sorted()`*

# In[59]:

def answer_four():
    dist = nltk.FreqDist(moby_tokens)
    rslt = [k for k in dist if len(k) > 5 and dist[k]>150]
    sorted_rslt = sorted(rslt)
    return sorted_rslt

answer_four()


# ### Question 5
# 
# Find the longest word in text1 and that word's length.
# 
# *This function should return a tuple `(longest_word, length)`.*

# In[51]:

def answer_five():
    word_set = set(moby_tokens)
    max_len_word = max(word_set, key=len)
    return (max_len_word, len(max_len_word))

answer_five()


# ### Question 6
# 
# What unique words have a frequency of more than 2000? What is their frequency?
# 
# "Hint:  you may want to use `isalpha()` to check if the token is a word and not punctuation."
# 
# *This function should return a list of tuples of the form `(frequency, word)` sorted in descending order of frequency.*

# In[47]:

def answer_six():
    dist = nltk.FreqDist(moby_tokens)
    sorted_dist = [( dist[k], k) for k in sorted(dist, reverse = True, key = dist.get ) if k.isalpha() and dist[k]>2000]
    
    
    return sorted_dist

answer_six()


# ### Question 7
# 
# What is the average number of tokens per sentence?
# 
# *This function should return a float.*

# In[37]:

def answer_seven():
    sentences = nltk.sent_tokenize(moby_raw)
    
    return len(text1)/len(sentences)

answer_seven()


# ### Question 8
# 
# What are the 5 most frequent parts of speech in this text? What is their frequency?
# 
# *This function should return a list of tuples of the form `(part_of_speech, frequency)` sorted in descending order of frequency.*

# In[46]:

def answer_eight():
    set_words = set(text1)
    
    pos_tags = nltk.pos_tag(text1)
    rslt = {}
    for x,y in pos_tags:
        if y in rslt:
            rslt[y] += 1
        else:
            rslt[y] = 1
    
    val_rslt = [(k,rslt[k]) for k in sorted(rslt, reverse = True, key = rslt.get)]
    
    
    return val_rslt[:5]

answer_eight()


# ## Part 2 - Spelling Recommender
# 
# For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.
# 
# For every misspelled word, the recommender should find find the word in `correct_spellings` that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.
# 
# *Each of the three different recommenders will use a different distance measure (outlined below).
# 
# Each of the recommenders should provide recommendations for the three default words provided: `['cormulent', 'incendenece', 'validrate']`.

# In[44]:

from nltk.corpus import words

correct_spellings = words.words()


# ### Question 9
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the trigrams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[69]:

def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    c = [i for i in correct_spellings if i[0]=='c']
    x = [(nltk.jaccard_distance(set(nltk.ngrams(entries[0], n=3)), 
                                  set(nltk.ngrams(a, n=3))), a) for a in c]
    sorted_one = sorted(x)
    
    inc = [i for i in correct_spellings if i[0]=='i']
    y = [(nltk.jaccard_distance(set(nltk.ngrams(entries[1], n=3)), 
                                  set(nltk.ngrams(a, n=3))), a) for a in inc]
    sorted_two = sorted(y)
    
    v = [i for i in correct_spellings if i[0]=='v']
    z = [(nltk.jaccard_distance(set(nltk.ngrams(entries[2], n=3)), 
                                  set(nltk.ngrams(a, n=3))), a) for a in v]
    sorted_three = sorted(z)
    
    
    
    
    return [sorted_one[0][1], sorted_two[0][1], sorted_three[0][1]]
    
answer_nine()


# ### Question 10
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the 4-grams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[68]:

def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    c = [i for i in correct_spellings if i[0]=='c']
    x = [(nltk.jaccard_distance(set(nltk.ngrams(entries[0], n=4)), 
                                  set(nltk.ngrams(a, n=4))), a) for a in c]
    sorted_one = sorted(x)
    
    inc = [i for i in correct_spellings if i[0]=='i']
    y = [(nltk.jaccard_distance(set(nltk.ngrams(entries[1], n=4)), 
                                  set(nltk.ngrams(a, n=4))), a) for a in inc]
    sorted_two = sorted(y)
    
    v = [i for i in correct_spellings if i[0]=='v']
    z = [(nltk.jaccard_distance(set(nltk.ngrams(entries[2], n=4)), 
                                  set(nltk.ngrams(a, n=4))), a) for a in v]
    sorted_three = sorted(z)
    
    
    
    
    return [sorted_one[0][1], sorted_two[0][1], sorted_three[0][1]]
    
answer_ten()


# ### Question 11
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Edit distance on the two words with transpositions.](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[67]:

def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    c = [i for i in correct_spellings if i[0]=='c']
    x = [(nltk.edit_distance(entries[0], a), a) for a in c]
    
    c = [i for i in correct_spellings if i[0]=='i']
    y = [(nltk.edit_distance(entries[1], a), a) for a in c]
    
    c = [i for i in correct_spellings if i[0]=='v']
    z = [(nltk.edit_distance(entries[2], a), a) for a in c]
    
    return [sorted(x)[0][1], sorted(y)[0][1], sorted(z)[0][1]]
    
answer_eleven()


# In[ ]:



