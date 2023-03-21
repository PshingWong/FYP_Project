import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from PIL import Image
import pyLDAvis.sklearn
st.set_page_config(
    page_icon="📝",
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

st.title('Title & Content Analysis')


#Create Title & Content page
# Create two columns
col17, col18 = st.columns(2)

# Data visualization for the sentiment analysis result of title in en_False.csv
df_en_title = df_en.groupby('sentiment_title').count()
col17.write("Sentiment Analysis of Title in en_False.csv")
col17.bar_chart(df_en_title['title'])

# Data visualization for the sentiment analysis result of Content in en_False.csv
df_en_content = df_en.groupby('sentiment_content').count()
col18.write("Sentiment Analysis of Content in en_False.csv")
col18.bar_chart(df_en_content['content_text'])
st.write("<h5 style='text-align: left;'> ??? </h5>", unsafe_allow_html=True)

# Create two columns
col19, col20 = st.columns(2)

# Data visualization for the sentiment analysis result of title in es_False.csv
df_es_title = df_es.groupby('sentiment_title').count()
col19.write("Sentiment Analysis of Title in es_False.csv")
col19.bar_chart(df_es_title['source_title_translated'])

# Data visualization for the sentiment analysis result of Content in es_False.csv
df_es_content = df_es.groupby('sentiment_content').count()
col20.write("Sentiment Analysis of Content in es_False.csv")
col20.bar_chart(df_es_content['content_text_translated'])
st.write("<h5 style='text-align: left;'> ??? </h5>", unsafe_allow_html=True)

# Create two columns
col21, col22 = st.columns(2)

# Data visualization for the sentiment analysis result of title in pt_False.csv
df_pt_title = df_pt.groupby('sentiment_title').count()
col21.write("Sentiment Analysis of Title in pt_False.csv")
col21.bar_chart(df_pt_title['source_title_translated'])

# Data visualization for the sentiment analysis result of Content in pt_False.csv
df_pt_content = df_pt.groupby('sentiment_content').count()
col22.write("Sentiment Analysis of Content in pt_False.csv")
col22.bar_chart(df_pt_content['content_text_translated'])
st.write("<h5 style='text-align: left;'> ??? </h5>", unsafe_allow_html=True)

# Create two columns
col23, col24 = st.columns(2)

# Data visualization for the sentiment analysis result of title in fr_False.csv
df_fr_title = df_fr.groupby('sentiment_title').count()
col23.write("Sentiment Analysis of Title in fr_False.csv")
col23.bar_chart(df_fr_title['source_title_translated'])

# Data visualization for the sentiment analysis result of Content in fr_False.csv
df_fr_content = df_fr.groupby('sentiment_content').count()
col24.write("Sentiment Analysis of Content in fr_False.csv")
col24.bar_chart(df_fr_content['content_text_translated'])
st.write("<h5 style='text-align: left;'> ??? </h5>", unsafe_allow_html=True)

