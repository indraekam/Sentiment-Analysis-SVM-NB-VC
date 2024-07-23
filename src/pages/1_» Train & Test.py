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

st.set_page_config(page_title="Sentiment Analysis", page_icon="📄", layout="wide")

tab1 = st.container()


# FUNCTION


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

def cleaning(df):
    for i, r in df.iterrows():
        y = clean_tweet(r['tweet'])
        df.loc[i, 'tweet_cleaned'] = y

    df.drop_duplicates(subset='tweet_cleaned', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv('tweet_cleaned.csv', encoding='utf-8', index=False)
    st.session_state.df1 = df

def word_tokenize_wrapper(text):
    return word_tokenize(text)

def untokenize(data):
    for tokens in data:
        yield ' '.join(map(str, tokens))

def tokenization(df):
    nltk.download('punkt')
    df['tweet_tokens'] = df['tweet_cleaned'].apply(word_tokenize_wrapper)
    df.to_csv('tweet_tokenized.csv', index=False)
    st.session_state.df2 = df

def normalization(df):
    dataslang1 = pd.read_csv('colloquial-indonesian-lexicon.csv', usecols=['slang', 'formal'])
    dataslang2 = pd.read_csv('formalizationDict.txt', delimiter = "\t", names=["slang", "formal"])
    dataslang = pd.concat([dataslang1, dataslang2], ignore_index=True)

    dataslang.drop_duplicates(subset='slang', inplace=True)
    dataslang.reset_index(drop=True, inplace=True)
    dataslang.to_csv('combine_normalization.csv', index=False)

    data_slang = dataslang['slang'].to_list()
    token_tweets = df['tweet_tokens']

    tweets = []
    for data in token_tweets:
        tweet = []
        for i in range(len(data)):
            if data[i] in data_slang:
                indexslang = data_slang.index(data[i])
                tweet.append(dataslang['formal'][indexslang])
            else:
                tweet.append(data[i])
        tweets.append(tweet)
    df['normalization'] = tweets
    df['normalization'] = list(untokenize(df['normalization']))
    df['normalization'] = df['normalization'].apply(word_tokenize_wrapper)
    df.to_csv('tweet_normalization.csv', encoding='utf-8', index=False)
    st.session_state.df3 = df

def stopwords_removal(words):
    stop_factory = StopWordRemoverFactory()
    more_stopword = ['uu', 'ruu', 'rancangan', 'ruutpks', 'undang', 'tpks', 'uutpks', 'tindak', 'pidana', 
                 'kekerasan', 'seksual', 'korban', 'kejahatan', 'pelecehan', 'pemaksaan', 'rapat', 'penjara', 'diancam', 
                 'ancaman', 'pasal', 'implementasi', 'diimplementasikan', 'mengimplementasikan', 'aborsi', 
                 'pahlawan', 'puan', 'maharani', 'dpr', 'indonesia', 'perempuan', 'memperjuangkan', 'diperjuangkan', 
                 'perjuangan', 'perjuangannya', 'ri', 'berikut',  'atas', 'sih', 'apa-apa', 'amp', 'aa', 'iya', 'si', 'eh', 'kak', 'oh', 
                 'he', 'nder']
    
    list_stopwords = stop_factory.get_stop_words()+more_stopword
    return [word for word in words if word not in list_stopwords]

def stopwords(df):
    df['tweet_stopword'] = df['normalization'].apply(stopwords_removal) 
    df.to_csv('tweet_stopwords.csv', encoding='utf-8', index=False)
    st.session_state.df4 = df

def stemming_on_text(data):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = [stemmer.stem(word) for word in data]
    return text

def stemming(df):
    df['tweet_stemming']= df['tweet_stopword'].apply(lambda x: stemming_on_text(x))
    df['final_tweet'] = list(untokenize(df['tweet_stemming']))
    st.session_state.df5 = df

def analyze(score):
    if score > 0:
        return 'Positive'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Negative'

def labelling(df):
    path_neg = open('InSet/negative.tsv') 
    path_pos = open('InSet/positive.tsv') 
    lexicon_neg = pd.read_csv(path_neg, sep='\t')
    lexicon_pos = pd.read_csv(path_pos, sep='\t')
    lexicon_neg_word = lexicon_neg['word'].to_list()
    lexicon_pos_word = lexicon_pos['word'].to_list()

    tweets = df['tweet_stemming']

    # COUNT NEG SCORE

    neg_score = []
    for data in tweets:
        sentiment = 0
        for i in range(len(data)):
            if str(data[i]) in lexicon_neg_word:
                print(data[i])
                sentiment += int(lexicon_neg['weight'][lexicon_neg_word.index(data[i])])
            else:
                sentiment += 0
        print(sentiment)
        neg_score.append(sentiment)
    
    # COUNT POS SCORE

    pos_score = []
    for data in tweets:
        sentiment = 0
        for i in range(len(data)):
            if str(data[i]) in lexicon_pos_word:
                print(data[i])
                sentiment += int(lexicon_pos['weight'][lexicon_pos_word.index(data[i])])
            else:
                sentiment += 0
        print(sentiment)
        pos_score.append(sentiment)

    df['neg_score'] = neg_score
    df['pos_score'] = pos_score
    df['polarity'] = df['neg_score'] + df['pos_score']
    df['sentiment'] = df['polarity'].apply(analyze)
    st.session_state.df6 = df

def eda(df):

    # most common words in each sentiment

    Positive_sent = df[df['sentiment']=='Positive']
    Negative_sent = df[df['sentiment']=='Negative']
    Neutral_sent = df[df['sentiment']=='Neutral']

    # Most common positive words

    st.session_state.top1 = Counter([item for sublist in Positive_sent['tweet_stemming'] for item in sublist])
    st.session_state.temp_positive = pd.DataFrame(st.session_state.top1.most_common(30))
    st.session_state.wordcloud_pos = WordCloud(background_color="white").generate(str(st.session_state.temp_positive))

    # Most common neutral words

    st.session_state.top2 = Counter([item for sublist in Neutral_sent['tweet_stemming'] for item in sublist])
    st.session_state.temp_neutral = pd.DataFrame(st.session_state.top2.most_common(30))
    st.session_state.wordcloud_neu = WordCloud(background_color="white").generate(str(st.session_state.temp_neutral))

    
    # Most common negative words

    st.session_state.top3 = Counter([item for sublist in Negative_sent['tweet_stemming'] for item in sublist])
    st.session_state.temp_negative = pd.DataFrame(st.session_state.top3.most_common(30))
    st.session_state.wordcloud_neg = WordCloud(background_color="white").generate(str(st.session_state.temp_negative))


def classification_all(df):
    x = df.final_tweet
    y = df.sentiment
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10, random_state=None)
    st.session_state.vectorizer = TfidfVectorizer(max_features=1000)
    st.session_state.accuracy_nb = []
    st.session_state.accuracy_svm = []
    st.session_state.accuracy_sv = []

    i = 0

    for train_index, test_index in kf.split(x):

        i += 1

        X_train, X_test = x[train_index], x[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        
        Xtrain = st.session_state.vectorizer.fit_transform(X_train)   
        Xtest = st.session_state.vectorizer.transform(X_test)
        
        st.session_state.clf1 = MultinomialNB(alpha=0.01)
        st.session_state.clf1.fit(Xtrain, Y_train)
        
        st.session_state.clf2 = SVC(C=4, kernel='linear', probability=True, random_state=7)
        st.session_state.clf2.fit(Xtrain, Y_train)
        
        st.session_state.clf3 = VotingClassifier(estimators=[('mnb', st.session_state.clf1), ('svm', st.session_state.clf2)], voting='soft', weights=[1, 3])
        st.session_state.clf3.fit(Xtrain, Y_train)

        if i == 3:
            st.session_state.sv_best_model = st.session_state.clf3

        ypred_nb = st.session_state.clf1.predict(Xtest)
        st.session_state.accuracy_nb.append(accuracy_score(Y_test, ypred_nb))
        
        ypred_svm = st.session_state.clf2.predict(Xtest)
        st.session_state.accuracy_svm.append(accuracy_score(Y_test, ypred_svm))
        
        ypred_sv = st.session_state.clf3.predict(Xtest)
        st.session_state.accuracy_sv.append(accuracy_score(Y_test, ypred_sv))

# TAB UPLOAD DATA  

tab1.subheader("Upload CSV of data tweets")

tab1.uploaded_file = tab1.file_uploader("Choose a file", type='csv')
tab1.btn = tab1.container()
with tab1.btn:
    if tab1.button('Upload'):
        if tab1.uploaded_file is not None:
            tab1.msg_placehoder = tab1.empty()
            tab1.my_bar = tab1.progress(0)
            st.session_state.percent = 0
            tab1.msg_placehoder.success("Upload dataset in process, please wait....")
            tab1.df = pd.read_csv(tab1.uploaded_file, usecols=['datetime', 'username', 'tweet'])
            st.session_state.df = tab1.df
            st.session_state.df['datetime'] = pd.to_datetime(st.session_state.df['datetime'])
            st.session_state.df.sort_values(by='datetime', inplace=True)
            st.session_state.df.reset_index(drop=True, inplace=True)
            if 'df' in st.session_state:
                st.session_state.percent += 10
                tab1.my_bar.progress(st.session_state.percent)
            
            # Cleaning
            tab1.msg_placehoder.success("Cleaning dataset in process, please wait....")
            cleaning(st.session_state.df)
            if 'df1' in st.session_state:
                st.session_state.percent += 10
                tab1.my_bar.progress(st.session_state.percent)
            
            tab1.msg_placehoder.success("Tokenization in process, please wait....")
            tokenization(st.session_state.df1)
            if 'df2' in st.session_state:
                st.session_state.percent += 10
                tab1.my_bar.progress(st.session_state.percent)
                
            tab1.msg_placehoder.success("Normalization in process, please wait....")
            normalization(st.session_state.df2)
            if 'df3' in st.session_state:
                st.session_state.percent += 10
                tab1.my_bar.progress(st.session_state.percent)
                
            tab1.msg_placehoder.success("Stop words removal in process, please wait....")
            stopwords(st.session_state.df3)
            if 'df4' in st.session_state:
                st.session_state.percent += 10
                tab1.my_bar.progress(st.session_state.percent)

            tab1.msg_placehoder.success("Stemming in process, please wait....")
            stemming(st.session_state.df4)
            if 'df4' in st.session_state:
                st.session_state.percent += 10
                tab1.my_bar.progress(st.session_state.percent)

            tab1.msg_placehoder.success("Labelling in process, please wait....")
            labelling(st.session_state.df5)
            if 'df4' in st.session_state:
                st.session_state.percent += 10
                tab1.my_bar.progress(st.session_state.percent)
            
            tab1.msg_placehoder.success("Exploratory Data Analysis in process, please wait....")
            eda(st.session_state.df6)
            if 'df4' in st.session_state:
                st.session_state.percent += 10
                tab1.my_bar.progress(st.session_state.percent)
            
            tab1.msg_placehoder.success("Classification in process, please wait....")
            classification_all(st.session_state.df6)
            if 'accuracy_sv' in st.session_state:
                st.session_state.percent += 20
                tab1.my_bar.progress(st.session_state.percent)
                tab1.msg_placehoder.success("Done")
                