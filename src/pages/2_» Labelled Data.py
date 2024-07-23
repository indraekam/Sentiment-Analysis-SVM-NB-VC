import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Labelled Data", page_icon="📄", layout="wide")

st.markdown("# Labelled Data")
data = st.container()

with data:
    # df = pd.DataFrame(
    # np.random.randn(50, 20),
    # columns=('col %d' % i for i in range(20)))
    if 'df5' in st.session_state:
        st.dataframe(st.session_state.df5, use_container_width = True)  # Same as st.write(df)
