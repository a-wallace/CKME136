#!/usr/bin/env python
# coding: utf-8

# # Carbon Tax Sentiment Analysis - Initial Code and Results
# 
# To begin the creation of a twitter sentiment analyzer, there are four steps that must be accomplished:
# 
# 1. Importing libraries and dataset
# 2. Initial Data Cleaning
# 3. Dividing data into Training and test sets
# 4. Model prediction and evaluation
# 
# After these steps have been accomplished, an initil starting point will be established from which the model can be improved by experimenting with different pre-processing techniques and machine learning algorithms. 
# 
# 

# ## Step 1: Importing Libraries and Dataset
# 

# In[1]:


# import required libraries

import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
'exec(%matplotlib inline)'

# import dataset

data_source_url = "carbon_tax_tweets.csv"
carbon_tweets = pd.read_csv(data_source_url)

# ensure nltk stopword database is present
nltk.download('stopwords')


# ## Step 2: Initial Data Cleaning

# In[2]:


# DATA CLEANING 

# remove all special characters
carbon_tweets['Tweet'] =  [re.sub(r'\W', ' ', str(x)) for x in carbon_tweets['Tweet']]

# remove all single characters
carbon_tweets['Tweet'] =  [re.sub(r'\+[a-zA-Z]\s+', ' ', str(x)) for x in carbon_tweets['Tweet']]

# remove single characters from the start
carbon_tweets['Tweet'] =  [re.sub(r'\^[a-zA-Z]\s+', ' ', str(x)) for x in carbon_tweets['Tweet']]

# substituting multiple spaces with single space
carbon_tweets['Tweet'] =  [re.sub(r'\s+', ' ', str(x)) for x in carbon_tweets['Tweet']]

# removing prefixed 'b'
carbon_tweets['Tweet'] =  [re.sub(r'^b\s+', ' ', str(x)) for x in carbon_tweets['Tweet']]

# converting to lowercase
carbon_tweets['Tweet'] =  [x.lower() for x in carbon_tweets['Tweet']]


# ## Step 3: Dividing data into Training and Test sets

# In[3]:


# divide data into training and tests sets	

dependent_vars = carbon_tweets['Tweet']
independent_vars = carbon_tweets["Polarity"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dependent_vars, independent_vars, test_size=0.2, random_state=0)

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
training_data_transformed = vectorizer.fit_transform(X_train)
testing_data_transformed = vectorizer.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(training_data_transformed, y_train)


# ## Step 4: Model prediction and evaluation

# In[4]:


# make predictions / evaluating model

predictions = text_classifier.predict(testing_data_transformed)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, predictions))
print("CLASSIFICATION REPORT:")
print(classification_report(y_test, predictions))
print("ACCURACY SCORE:")
print(accuracy_score(y_test, predictions))


# ## Conclusion and Next Steps
# 
# 
# The model performs poorly with an accuracy of 50%. This is to be expected, given the extremely small number of observations present in the dataset and the minimal text pre-processing that took place. There is high potential to vastly improve the accuracy of the model by doing three things. First, increase the dataset size. I have figured out the data limitation the Twitter API places on users and will implement a daatset for the final project with 3,000 observations. This will strenghten the testing and training data. Second, implement more advanced text pre-processing steps learned from the literature review. 
# Third, experiment with different NLP machine learning algorithms. Random Forest was used here, but it will be interesting to see the difference in accuracy when Naive Bayes is used or some other algorithm.
