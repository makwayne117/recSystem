#we want three users to try this out on

from letterboxdpy import user
from letterboxdpy import movie
import requests 
from bs4 import BeautifulSoup 
import sqlite3
import pandas as pd

from SGD import *

def return_reviews():
    path = "IMDB Dataset.csv"
    data = pd.read_csv(path)
    reviews = data['review']
    sentiments = data['sentiment']

    sentiment_values = []

    for s in sentiments:
        if s == 'positive':
            sentiment_values.append(1)
        else:
            sentiment_values.append(0)
        

    #print(reviews)
    print(sentiments)

    return list(zip(reviews,sentiment_values))

data = return_reviews()

#split data into training and validation
#DELETE EVERYTHING BELOW
random.shuffle(data)  # Shuffle for a more realistic training scenario

# Split data into training and validation sets
split_index = int(len(data) * 0.8)
train_data = data[:split_index]
validation_data = data[split_index:]

# Define parameters
num_epochs = 10
learning_rate = 0.01

# Train the predictor
weights = learnPredictor(train_data, validation_data, extractWordFeatures, num_epochs, learning_rate)

print("Learned weights:", weights)


    #print(data.head())