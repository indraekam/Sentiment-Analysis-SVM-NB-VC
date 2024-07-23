import streamlit as st
# import time
from datetime import timedelta
# import snscrape.modules.twitter as sntwitter
import pandas as pd
import csv
import regex as re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
# import plotly.express as px
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
st.set_page_config(page_title="Accuracy", page_icon="📚", layout="wide")

st.markdown("# Predict")

st.input = st.container()
st.btn = st.container()

# function

def clean_tweet(tweet):
    # CASE FOLDING
    c1 = tweet.lower()
    
    #REMOVE PUNCTUATION
    # Replace RT tag
    c2 = re.sub('RT\s', '', c1)
    
    # Replace @_username
    c3 = re.sub('@[A-Za-z0-9_]+','', c2)
    
    # Replace URL
    c4 = re.sub('(http|https):\/\/\S+', '', c3)
    
    # Replace hashtag
    c5 = re.sub('#[A-Za-z0-9_]+','', c4)
    
    # Replace word repetition with a single occurance ('ooooooooo' become 'oo')
    c6 = re.sub(r'(.)\1+', r'\1\1', c5)
    
    # Alphabets only, exlude numbers and special characters
    c7 = re.sub(r'[^a-zA-Z]', ' ', c6)
    
    return c7

def cleaning(tweet):
    y = clean_tweet(tweet)
    y.drop_duplicates(subset='tweet_cleaned', inplace=True)
    y.reset_index(drop=True, inplace=True)
    return y

def word_tokenize_wrapper(text):
    # nltk.download('punkt')
    return word_tokenize(text)

def untokenize(data):
    for tokens in data:
        tweet = yield ' '.join(map(str, tokens))
    return tweet

def normalization(tweet):
    dataslang = pd.read_csv('combine_normalization.csv', index=False)
    data_slang = dataslang['slang'].to_list()
    tweet_normal = []
    for i in range(len(tweet)):
        if tweet[i] in data_slang:
            indexslang = data_slang.index(tweet[i])
            tweet_normal.append(dataslang['formal'][indexslang])
        else:
            tweet_normal.append(tweet[i])

    tweet_normal = untokenize(tweet_normal)
    return tweet_normal

def stopwords_removal(words):
    stop_factory = StopWordRemoverFactory()
    more_stopword = ['iya', 'sih', 'apa-apa', 'korban', 'kekerasan', 'kejahatan', 'pelecehan', 'pemaksaan', 'rapat', 
                 'tpks', 'uu', 'tindak', 'pidana', 'puan', 'maharani', 'ruu', 'undang', 'undang-undang', 'uutpks', 'seksual', 
                 'berikut', 'pahlawan', 'perempuan', 'aborsi', 'menjabat', 'pengesahan', 'mengusahakan', 'dibawa']
    
    list_stopwords = stop_factory.get_stop_words()+more_stopword
    return [word for word in words if word not in list_stopwords]

def stopwords(tweet):
    tw = stopwords_removal(tweet) 
    return tw

def stemming_on_text(data):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = [stemmer.stem(word) for word in data]
    return text

def stemming(tweet):
    tw = stemming_on_text(tweet)
    return tw

with st.input:
    if 'clf3' in st.session_state:
        text = st.text_input(
            "Input text to predict sentiment"
        )
with st.btn:
    if 'clf3' in st.session_state:
        if st.button('Predict'):
            st.msg_placehoder = st.empty()
            if len(text) <= 0:
                st.msg_placehoder.error("Please input text")
            else:
                text=[text]
                tweet_vec = st.session_state.vectorizer.transform(text)
                # ypred_nb = st.session_state.clf1.predict(tweet_vec)
                # ypred_svm = st.session_state.clf2.predict(tweet_vec)
                ypred_sv = st.session_state.sv_best_model.predict(tweet_vec)
                # st.write("Multinomial Naive Bayes: ", str(ypred_nb))
                # st.write("Support Vector Machine: ", str(ypred_svm))
                for i in ypred_sv:
                    st.info('Sentiment: ' + i)
