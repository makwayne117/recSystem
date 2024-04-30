from SGD import *

import requests 
import pandas as pd
import numpy as np

df = pd.read_csv('IMDB Dataset.csv')

df['sentiment'] = df['sentiment'].replace({'positive': 1, 'negative': 0})

df['review'] = df['review'].str.replace("isn't", 'is not')

negation = {'not good': 'notgood', 'not bad': 'notbad', 'not great': 'notgreat'}
df['review'] = df['review'].replace(negation, regex=True)

features = ['good', 'bad', 'amazing', 'okay', 'terrible', 'solid', 'poor', 'decent', 
            'great', 'notgood', 'notbad', 'notgreat']

feature_vector = pd.DataFrame()

for feature in features:
    feature_vector[feature] = df['review'].str.count(feature)

target = df['sentiment']

np.random.seed(42) 
mask = np.random.rand(len(df)) < 0.7
train_data = feature_vector[mask]
test_data = feature_vector[~mask]
train_labels = target[mask]
test_labels = target[~mask]


X_train = train_data.values
y_train = train_labels.values
X_test = test_data.values
y_test = test_labels.values
