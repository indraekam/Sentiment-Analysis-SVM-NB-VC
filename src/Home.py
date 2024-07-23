import streamlit as st
from PIL import Image

st.set_page_config(page_title="Home", page_icon=":home:", layout="wide")

st.markdown("<h1 style='text-align: center;'>ANALISIS SENTIMEN TERHADAP UU TPKS DI TWITTER MENGGUNAKAN INSET LEXICON DENGAN ALGORITMA MNB DAN SVM BERBASIS SOFT VOTING</h1>", unsafe_allow_html=True)
col1, col2, col3 = st.columns((3,6,3))

with col1:
    st.write(' ')
with col2:
    st.image("sentiment.png")
with col3:
    st.write(' ')

