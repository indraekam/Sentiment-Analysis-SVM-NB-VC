import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Exploratory Data Analysis", page_icon="📊", layout="wide")

st.markdown("# Exploratory Data Analysis")

col1, col2 = st.columns(2)


with col1:
    if 'df6' in st.session_state:
        st.session_state.df6['datetime'] = st.session_state.df6['datetime'].apply(lambda a: pd.to_datetime(a).date()) 
        fig_date = plt.figure(figsize = (12, 6))
        bc_1 = sns.countplot(x = 'datetime', data = st.session_state.df6)
        bc_1.bar_label(bc_1.containers[0])
        #sns.countplot(x = 'datetime', data = st.session_state.df6)
        st.pyplot(fig_date)
with col2:
    if 'df6' in st.session_state:
        fig_sentiment = plt.figure(figsize = (12, 6))
        bc2 = sns.countplot(x = 'sentiment', data = st.session_state.df6)
        bc2.bar_label(bc2.containers[0])
        # sns.countplot(x = 'sentiment', data = st.session_state.df6)
        st.pyplot(fig_sentiment)

col3, col4, col5 = st.columns(3)
#st.col3 = st.container()
#st.col4 = st.container()
#st.col5 = st.container()

with col3:
    if 'temp_positive' in st.session_state:
        st.image(st.session_state.wordcloud_pos.to_array(), width=400, caption='Wordcloud Positive Words')
with col4:
    if 'temp_neutral' in st.session_state:
        st.image(st.session_state.wordcloud_neu.to_array(), width=400, caption='Wordcloud Neutral Words')

with col5:
    if 'temp_negative' in st.session_state:
        st.image(st.session_state.wordcloud_neg.to_array(), width=400, caption='Wordcloud Negative Words')