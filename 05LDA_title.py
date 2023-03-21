import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from gensim import matutils, models, corpora
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

st.set_page_config(
    page_icon="🧊",
    layout="wide",
)
st.markdown("""
<style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        width: 150px;
        color: black;
    }
    .Widget>label {
        color: black;
        font-size: 18px;
    }
    h1 {
        color: black;
        font-size: 64px;
    }
    h2 {
        color: black;
        font-size: 48px;
    }
    h3 {
        color: black;
        font-size: 36px;
    }
    h5 {
        color: black;
        font-size: 24px;
    }
    h5 {
        color: black;
        font-size: 18px;
    }
    .stRadio label {
        font-size: 60px;
    }
</style>
""", unsafe_allow_html=True)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

df_original = pd.read_csv('FakeCovid_July2020.csv')
df_en = pd.read_csv('en_False_tran.csv')
df_es = pd.read_csv('es_False_tran.csv')
df_pt = pd.read_csv('pt_False_tran.csv')
df_fr = pd.read_csv('fr_False_tran.csv')

st.sidebar.metric("Number of Fake News(en)",df_en.shape[0])
st.sidebar.metric("Number of Fake News(es)",df_es.shape[0])
st.sidebar.metric("Number of Fake News(pt)",df_pt.shape[0])
st.sidebar.metric("Number of Fake News(fr)",df_fr.shape[0])

# Display the HTML file in Streamlit
st.components.v1.html(open('en_title_visualization.html', 'r').read(), width=1400, height=900, scrolling=True)
st.components.v1.html(open('es_title_visualization.html', 'r').read(), width=1400, height=900, scrolling=True)
st.components.v1.html(open('fr_title_visualization.html', 'r').read(), width=1400, height=900, scrolling=True)
st.components.v1.html(open('pt_title_visualization.html', 'r').read(), width=1400, height=900, scrolling=True)