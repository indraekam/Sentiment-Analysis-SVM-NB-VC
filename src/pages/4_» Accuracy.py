import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Accuracy", page_icon="📚", layout="wide")

st.markdown("# Result")
# st.sidebar.header("Exploratory Data Analysis")


if 'accuracy_nb' in st.session_state:
    st.write("Multinomial Naive Bayes :" , format(np.mean(st.session_state.accuracy_nb)*100, ".2f"))

if 'accuracy_svm' in st.session_state:
    st.write("Support Vector Machine :" , format(np.mean(st.session_state.accuracy_svm)*100, ".2f"))

if 'accuracy_sv' in st.session_state:
    st.write("Soft Voting :" , format(np.mean(st.session_state.accuracy_sv)*100, ".2f"))

col1, col2, col3 = st.columns(3)

with col1:
    if 'accuracy_nb' in st.session_state:
        df1 = pd.DataFrame()
        df1['accuracy_mnb']=pd.Series(st.session_state.accuracy_nb)
        st.dataframe(df1, use_container_width = True)

with col2:
    if 'accuracy_svm' in st.session_state:
        df2 = pd.DataFrame()
        df2['accuracy_svm']=pd.Series(st.session_state.accuracy_svm)
        st.dataframe(df2, use_container_width = True)

with col3:
    if 'accuracy_sv' in st.session_state:
        df2 = pd.DataFrame()
        df2['accuracy_sv']=pd.Series(st.session_state.accuracy_sv)
        st.dataframe(df2, use_container_width = True)