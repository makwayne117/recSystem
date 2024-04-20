#we want three users to try this out on

from letterboxdpy import user
from letterboxdpy import movie
import requests 
from bs4 import BeautifulSoup 
import sqlite3
import pandas as pd

def return_reviews():
    path = "IMDB Dataset.csv"
    data = pd.read_csv(path)
    reviews = data['review']
    sentiments = data['sentiment']

    return reviews,sentiments

    #print(data.head())