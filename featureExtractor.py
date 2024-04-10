from senticnet.senticnet import SenticNet

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from textblob import TextBlob

import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import wordnet


#likely will need a better library. 

# TODO: Add the reviews section to get the reviews
# Code adapted from here: https://npogeant.medium.com/python-letterboxd-build-a-content-filtering-rec-sys-from-scratch-7648b25bccdc
async def fetch(url, session, input_data={}):
    async with session.get(url) as r:
        response = await r.read()

        # Parse ratings page response for each rating/review, use lxml parser for speed
        soup = BeautifulSoup(response, "lxml")
        
        movie_header = soup.find('section', attrs={'id': 'featured-film-header'})

        try:
            movie_title = movie_header.find('h1').text
        except AttributeError:
            movie_title = ''

        try:
            year = int(movie_header.find('small', attrs={'class': 'number'}).find('a').text)
        except AttributeError:
            year = None

        try:
            imdb_link = soup.find("a", attrs={"data-track-action": "IMDb"})['href']
            imdb_id = imdb_link.split('/title')[1].strip('/').split('/')[0]
        except:
            imdb_link = ''
            imdb_id = ''

        try:
            tmdb_link = soup.find("a", attrs={"data-track-action": "TMDb"})['href']
            tmdb_id = tmdb_link.split('/movie')[1].strip('/').split('/')[0]
        except:
            tmdb_link = ''
            tmdb_id = ''
        
        movie_object = {
                    "movie_id": input_data["movie_id"],
                    "movie_title": movie_title,
                    "year_released": year,
                    "imdb_id": imdb_id,
                    "tmdb_id": tmdb_id
                }

        return movie_object

def get_synonyms(word:str):
    synonyms = []
    
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name()) 
    
    return synonyms  

#adapted from: https://stackoverflow.com/questions/56980515/how-to-extract-all-adjectives-from-a-strings-of-text-in-a-pandas-dataframe
def extract_descriptive_words(sentence:str):
    blob = TextBlob(sentence)
    return [ word for (word,tag) in blob.tags if tag == "JJ"]


def constructFeature(sentence:str):
    vectors = {}
    adjectives = extract_descriptive_words(sentence)

    for s in adjectives:
        synonyms = get_synonyms(s)
        concept_info = sn.concept(s)
        polarity_label = sn.polarity_label(s)
        polarity_value = sn.polarity_value(s)
        moodtags = sn.moodtags(s)
        semantics = sn.semantics(s)
        sentics = sn.sentics(s)

        senticVector = {"synonyms": synonyms, "moodtags":moodtags, "polarity_label":polarity_label, "polarity_value":polarity_value, "semantics":semantics, "sentics":sentics}
        vectors[s] = senticVector
    return vectors

def generateReviewModel(title:str, sentence:str):
    vectors = {}

    senticVectors = constructFeature(sentence)

    return {"aspect":title, "category":"film quality", "review sentence": sentence, "sentic info":senticVectors}





