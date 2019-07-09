#import required libraries

import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
'exec(%matplotlib inline)'

#import dataset

data_source_url = r"out2.csv"
carbon_tweets = pd.read_csv(data_source_url)

nltk.download('stopwords')	# ensure nltk stopword database is present

## DATA CLEANING ##

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
	
##divide data into training and tests sets	

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

#make predictions / evaluating model

predictions = text_classifier.predict(testing_data_transformed)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, predictions))
print("CLASSIFICATION REPORT:")
print(classification_report(y_test, predictions))
print("ACCURACY SCORE:")
print(accuracy_score(y_test, predictions))

