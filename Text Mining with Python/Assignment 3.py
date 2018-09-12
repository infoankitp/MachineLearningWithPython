
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 3
# 
# In this assignment you will explore text message data and create models to predict if a message is spam or not. 

# In[4]:

import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)


# In[145]:

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)


# ### Question 1
# What percentage of the documents in `spam_data` are spam?
# 
# *This function should return a float, the percent value (i.e. $ratio * 100$).*

# In[146]:

def answer_one():
    spam_count = len(spam_data[spam_data['target'] == 1])
    total_count = len(spam_data['target'])
    
    return (spam_count/total_count)*100


# In[147]:

answer_one()


# ### Question 2
# 
# Fit the training data `X_train` using a Count Vectorizer with default parameters.
# 
# What is the longest token in the vocabulary?
# 
# *This function should return a string.*

# In[148]:

from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    vect = CountVectorizer().fit(X_train)
    vocab = vect.vocabulary_
    longest = sorted(vocab, reverse= True, key = len)
    return longest[0]


# In[149]:

answer_two()


# ### Question 3
# 
# Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.
# 
# Next, fit a fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`. Find the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[150]:

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():
    vect = CountVectorizer().fit(X_train)
    X_train_vect = vect.transform(X_train)
    X_test_vect = vect.transform(X_test)
    clf = MultinomialNB(alpha = 0.1).fit(X_train_vect, y_train)
    
    y_pred = clf.predict(X_test_vect)
    return roc_auc_score(y_test, y_pred)


# In[151]:

answer_three()


# ### Question 4
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer with default parameters.
# 
# What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?
# 
# Put these features in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series should be the feature name, and the data should be the tf-idf.
# 
# The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with largest tf-idfs should be sorted largest first. 
# 
# *This function should return a tuple of two series
# `(smallest tf-idfs series, largest tf-idfs series)`.*

# In[195]:

from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    vect = TfidfVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    feature_names = np.array(vect.get_feature_names())
    sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()
    small_index = feature_names[sorted_tfidf_index[:20]]
    small_value = X_train_vectorized.max(0).toarray()[0][sorted_tfidf_index[:20]]
    smallTuples = [(value, word) for word, value in zip(small_index, small_value)]
    smallTuples.sort()
    small_index = [element[1] for element in smallTuples]
    small_value = [element[0] for element in smallTuples]
    small_series = pd.Series(small_value,index=small_index)

    big_index = feature_names[sorted_tfidf_index[-20:]]
    big_value = X_train_vectorized.max(0).toarray()[0][sorted_tfidf_index[-20:]]
    bigTuples = [(-value, word) for word, value in zip(big_index, big_value)]
    bigTuples.sort()
    big_index = [element[1] for element in bigTuples]
    big_value = [-element[0] for element in bigTuples]
    big_series = pd.Series(big_value,index=big_index)
    return (small_series, big_series)


# In[196]:

answer_four()


# ### Question 5
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.
# 
# Then fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1` and compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[153]:

def answer_five():
    vect = TfidfVectorizer(min_df = 3).fit(X_train)
    X_train_vect = vect.transform(X_train)
    X_test_vect = vect.transform(X_test)
    clf = MultinomialNB(alpha = 0.1).fit(X_train_vect, y_train)
    
    y_pred = clf.predict(X_test_vect)
    
    return roc_auc_score(y_test, y_pred)


# In[154]:

answer_five()


# ### Question 6
# 
# What is the average length of documents (number of characters) for not spam and spam documents?
# 
# *This function should return a tuple (average length not spam, average length spam).*

# In[155]:

def answer_six():
    spam_data['len'] = spam_data['text'].str.len()
    spam_docs = spam_data[spam_data['target'] == 1]
    non_spam_docs = spam_data[spam_data['target'] == 0]
    spam_avg_len = spam_docs['len'].mean()
    non_spam_avg_len = non_spam_docs['len'].mean()
    
    return (non_spam_avg_len,spam_avg_len)


# In[156]:

answer_six()


# <br>
# <br>
# The following function has been provided to help you combine new features into the training data:

# In[157]:

def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# ### Question 7
# 
# Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5**.
# 
# Using this document-term matrix and an additional feature, **the length of document (number of characters)**, fit a Support Vector Classification model with regularization `C=10000`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[246]:

from sklearn.svm import SVC

def answer_seven():
    
    vect = TfidfVectorizer(min_df = 5).fit(X_train)
    X_train_vect = vect.transform(X_train)
    X_test_vect = vect.transform(X_test)
    X_train_final = add_feature(X_train_vect , X_train.str.len())
    X_test_final = add_feature(X_test_vect , X_test.str.len())
    clf = SVC(C = 10000).fit(X_train_final, y_train)
    
    y_pred = clf.predict(X_test_final)
    
    
    return roc_auc_score(y_test, y_pred)


# In[247]:

answer_seven()


# ### Question 8
# 
# What is the average number of digits per document for not spam and spam documents?
# 
# *This function should return a tuple (average # digits not spam, average # digits spam).*

# In[248]:

def answer_eight():
    def count_digits(x):
        count = 0
        for i in x:
            if i.isdigit():
                count +=1
            
        return count
    
    spam_data['dig_len'] = spam_data['text'].apply(count_digits)
    spam_docs = spam_data[spam_data['target'] == 1]
    non_spam_docs = spam_data[spam_data['target'] == 0]
    spam_avg_len = spam_docs['dig_len'].mean()
    non_spam_avg_len = non_spam_docs['dig_len'].mean()
    return (non_spam_avg_len, spam_avg_len)


# In[249]:

answer_eight()


# ### Question 9
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **word n-grams from n=1 to n=3** (unigrams, bigrams, and trigrams).
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * **number of digits per document**
# 
# fit a Logistic Regression model with regularization `C=100`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[267]:

from sklearn.linear_model import LogisticRegression

def answer_nine():
    def count_digits(x):
        count = 0
        for i in x:
            if i.isdigit():
                count +=1
            
        return count
    vect = TfidfVectorizer(min_df = 5, ngram_range = (1,3)).fit(X_train)
    X_train_vect = vect.transform(X_train)
    X_test_vect = vect.transform(X_test)
    X_train_final = add_feature(X_train_vect , X_train.str.len())
    X_train_final = add_feature(X_train_final, X_train.apply(count_digits))
    
    X_test_final = add_feature(X_test_vect , X_test.str.len())
    X_test_final = add_feature(X_test_final, X_test.apply(count_digits))
    
    
    clf = LogisticRegression(C = 100).fit(X_train_final, y_train)
    
    y_pred = clf.predict(X_test_final)
    
    
    return roc_auc_score(y_test, y_pred)


# In[268]:

answer_nine()


# ### Question 10
# 
# What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?
# 
# *Hint: Use `\w` and `\W` character classes*
# 
# *This function should return a tuple (average # non-word characters not spam, average # non-word characters spam).*

# In[252]:

def answer_ten():
    import re
    def non_word_len(x):
        count = 0
        for i in x:
            if re.match(r"\W",i):
                count +=1
        return count
    
    
    spam_data['non_word_len'] = spam_data['text'].apply(non_word_len)
    spam_docs = spam_data[spam_data['target'] == 1]
    non_spam_docs = spam_data[spam_data['target'] == 0]
    spam_avg_len = spam_docs['non_word_len'].mean()
    non_spam_avg_len = non_spam_docs['non_word_len'].mean()
    return (non_spam_avg_len, spam_avg_len)
    


# In[253]:

answer_ten()


# ### Question 11
# 
# Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **character n-grams from n=2 to n=5.**
# 
# To tell Count Vectorizer to use character n-grams pass in `analyzer='char_wb'` which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * number of digits per document
# * **number of non-word characters (anything other than a letter, digit or underscore.)**
# 
# fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# Also **find the 10 smallest and 10 largest coefficients from the model** and return them along with the AUC score in a tuple.
# 
# The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.
# 
# The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients:
# ['length_of_doc', 'digit_count', 'non_word_char_count']
# 
# *This function should return a tuple `(AUC score as a float, smallest coefs list, largest coefs list)`.*

# In[254]:

def answer_eleven():
    import re
    from sklearn.linear_model import LogisticRegression
    def count_digits(x):
        count = 0
        for i in x:
            if i.isdigit():
                count +=1
            
        return count
    def non_word_len(x):
        count = 0
        for i in x:
            if re.match(r"\W",i):
                count +=1
        return count
    vect = CountVectorizer(min_df = 5, ngram_range = (2,5), analyzer='char_wb').fit(X_train)
    X_train_vect = vect.transform(X_train)
    X_test_vect = vect.transform(X_test)
    X_train_final = add_feature(X_train_vect , X_train.str.len())
    X_train_final = add_feature(X_train_final, X_train.apply(count_digits))
    X_train_final = add_feature(X_train_final, X_train.apply(non_word_len))
    X_test_final = add_feature(X_test_vect , X_test.str.len())
    X_test_final = add_feature(X_test_final, X_test.apply(count_digits))
    X_test_final = add_feature(X_test_final, X_test.apply(non_word_len))
    
    clf = LogisticRegression(C = 100).fit(X_train_final, y_train)
    
    y_pred = clf.predict(X_test_final)
    feature_names = np.array(vect.get_feature_names())
    feature_names = np.concatenate((feature_names, np.array(['length_of_doc', 'digit_count', 'non_word_char_count'])))
    
    
    coefiecients = clf.coef_[0].argsort()
    
    auc_s = roc_auc_score(y_test, y_pred)
    low_coefs = list(feature_names[coefiecients[:10]])
    high_coefs = list(feature_names[coefiecients[-11:-1]])
    
    
    
    return (auc_s, low_coefs, high_coefs)


# In[255]:

answer_eleven()


# In[ ]:




# In[ ]:



