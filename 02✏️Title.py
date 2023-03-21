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
    page_icon="✏️",
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

st.title('Title Analysis')

# Create two columns
col1, col2 = st.columns(2)

# Data visualization for the sentiment analysis result of title in en_False.csv
df_en_title = df_en.groupby('sentiment_title').count()
col1.write("Sentiment Analysis of Title in en_False.csv")
col1.bar_chart(df_en_title['title'])

# Data visualization for the sentiment analysis result of title in en_False.csv sortby date
df_en_title = df_en.groupby(['published_date', 'sentiment_title']).count().reset_index()
df_en_title = df_en_title.rename(columns={'title': 'count'})
df_en_title = df_en_title.pivot(index='published_date', columns='sentiment_title', values='count').fillna(0)
df_en_title['date'] = df_en_title.index
df_en_title['date'] = pd.to_datetime(df_en_title['date'])
df_en_title['month'] = df_en_title['date'].dt.month
df_en_title = df_en_title.groupby('month').sum()
col2.write("Sentiment Analysis of Title in en_False.csv Sorted by Date")
col2.line_chart(df_en_title)
st.write("<h5 style='text-align: left;'> In the study of fake news about title analysis within the English language, the analysis reveals a distribution consisting of 388 cases with negative fake news, 345 cases with neutral fake news, and 317 cases exhibiting positive fake news. An examination of the publication dates indicates a notable upward trend in the proliferation of fake news during March 2020. This surge reached its zenith in April 2020, only to subsequently experience a decline commencing in May 2020. </h5>", unsafe_allow_html=True)
st.write("<h5 style='text-align: left;'> In the analysis of titles within the English language, the sentiment outcomes of three types are observed to be close number.</h5>", unsafe_allow_html=True)

# Create two columns
col3, col4 = st.columns(2)

# Data visualization for the sentiment analysis result of title in es_False.csv
df_es_title = df_es.groupby('sentiment_title').count()
col3.write("Sentiment Analysis of Title in es_False.csv")
col3.bar_chart(df_es_title['source_title_translated'])

# Data visualization for the sentiment analysis result of title in es_False.csv sortby date
df_es_title = df_es.groupby(['published_date', 'sentiment_title']).count().reset_index()
df_es_title = df_es_title.rename(columns={'source_title_translated': 'count'})
df_es_title = df_es_title.pivot(index='published_date', columns='sentiment_title', values='count').fillna(0)
df_es_title['date'] = df_es_title.index
df_es_title['date'] = pd.to_datetime(df_es_title['date'])
df_es_title['month'] = df_es_title['date'].dt.month
df_es_title = df_es_title.groupby('month').sum()
col4.write("Sentiment Analysis of Title in es_False.csv Sorted by Date")
col4.line_chart(df_es_title)
st.write("<h5 style='text-align: left;'> In the study of fake news about title analysis within the Spanish language, the analysis reveals a distribution consisting of 218 cases with negative fake news, 118 cases with neutral fake news, and 106 cases exhibiting positive fake news. An examination of the publication dates indicates a notable upward trend in the proliferation of fake news during March 2020. This surge reached its zenith in April 2020, only to subsequently experience a decline commencing in May 2020. At the same time negative fake news is rise up in February, and the amount of negative fake news surpassed the other two. </h5>", unsafe_allow_html=True)
st.write("<h5 style='text-align: left;'> In the analysis of titles within the Spanish language, the sentiment outcomes of three types showed more negative fake news. The number of fake news accounted for 42.5% of the tota, and positive fake news accounted for 20.7%, the lowest among them.Fake news articles written in Spain often skew negative. </h5>", unsafe_allow_html=True)

# Create two columns
col5, col6 = st.columns(2)

# Data visualization for the sentiment analysis result of title in pt_False.csv
df_pt_title = df_pt.groupby('sentiment_title').count()
col5.write("Sentiment Analysis of Title in pt_False.csv")
col5.bar_chart(df_pt_title['source_title_translated'])

# Data visualization for the sentiment analysis result of title in pt_False.csv sortby date
df_pt_title = df_pt.groupby(['published_date', 'sentiment_title']).count().reset_index()
df_pt_title = df_pt_title.rename(columns={'source_title_translated': 'count'})
df_pt_title = df_pt_title.pivot(index='published_date', columns='sentiment_title', values='count').fillna(0)
df_pt_title['date'] = df_pt_title.index
df_pt_title['date'] = pd.to_datetime(df_pt_title['date'])
df_pt_title['month'] = df_pt_title['date'].dt.month
df_pt_title = df_pt_title.groupby('month').sum()
col6.write("Sentiment Analysis of Title in pt_False.csv Sorted by Date")
col6.line_chart(df_pt_title)
st.write("<h5 style='text-align: left;'> In the study of fake news about title analysis within the Portuguese language, the analysis reveals a distribution consisting of 132 cases with negative fake news, 76 cases with neutral fake news, and 33 cases exhibiting positive fake news. An examination of the publication dates indicates a notable upward trend in the proliferation of fake news during March 2020. This surge reached its zenith in April 2020, only to subsequently experience a decline commencing in May 2020. At the same time negative fake news number is higher than the other two types of fake news from April. </h5>", unsafe_allow_html=True)
st.write("<h5 style='text-align: left;'> In the analysis of titles within the Portuguese language, the sentiment outcomes of three types showed more negative fake news. The number of fake news accounted for 54.7% of the tota, and positive fake news accounted for 13.6%, the lowest among them.From April alone, negative fake news outnumbered positive fake news by about 5 times.Fake news articles written in Portuguese often skew negative. </h5>", unsafe_allow_html=True)

# Create two columns
col7, col8 = st.columns(2)

# Data visualization for the sentiment analysis result of title in fr_False.csv
df_fr_title = df_fr.groupby('sentiment_title').count()
col7.write("Sentiment Analysis of Title in fr_False.csv")
col7.bar_chart(df_fr_title['source_title_translated'])

# Data visualization for the sentiment analysis result of title in fr_False.csv sortby date
df_fr_title = df_fr.groupby(['published_date', 'sentiment_title']).count().reset_index()
df_fr_title = df_fr_title.rename(columns={'source_title_translated': 'count'})
df_fr_title = df_fr_title.pivot(index='published_date', columns='sentiment_title', values='count').fillna(0)
df_fr_title['date'] = df_fr_title.index
df_fr_title['date'] = pd.to_datetime(df_fr_title['date'])
df_fr_title['month'] = df_fr_title['date'].dt.month
df_fr_title = df_fr_title.groupby('month').sum()
col8.write("Sentiment Analysis of Title in fr_False.csv Sorted by Date")
col8.line_chart(df_fr_title)
st.write("<h5 style='text-align: left;'> In the study of fake news about title analysis within the French language, the analysis reveals a distribution consisting of 81 cases with negative fake news, 41 cases with neutral fake news, and 37 cases exhibiting positive fake news. An examination of the publication dates indicates a notable upward trend in the proliferation of fake news during March 2020. This surge reached its zenith in April 2020, only to subsequently experience a decline commencing in May 2020. At the same time negative fake news number is higher than the other two types of fake news from April. </h5>", unsafe_allow_html=True)
st.write("<h5 style='text-align: left;'> In the analysis of titles within the French language, the sentiment outcomes of three types showed more negative fake news. The number of fake news accounted for 50.9% of the tota, and positive fake news accounted for 23.3%, the lowest among them.Fake news articles written in French often skew negative. </h5>", unsafe_allow_html=True)

st.write("<h3 style='text-align: left;'> Summarize </h3>", unsafe_allow_html=True)
st.write("<h5 style='text-align: left;'> Through the analysis of the Title, except for the fake news articles written in English, the articles written in most languages are negative updates, accounting for half of the number of this category.</h5>", unsafe_allow_html=True)
