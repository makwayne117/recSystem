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
#run the trainiing weights.
#test the weights with a test review to see if it's positive, negative, etc. 


    #print(data.head())