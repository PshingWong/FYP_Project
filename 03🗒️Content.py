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
    page_icon="🗒️",
    layout="wide",
    page_title="Content"

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

st.title('Content Analysis')

# Create two columns
col9, col10 = st.columns(2)

# Data visualization for the sentiment analysis result of Content in en_False.csv
df_en_content = df_en.groupby('sentiment_content').count()
col9.write("Sentiment Analysis of Content in en_False.csv")
col9.bar_chart(df_en_content['content_text'])

# Data visualization for the sentiment analysis result of Content in en_False.csv sortby date
df_en_content = df_en.groupby(['published_date', 'sentiment_content']).count().reset_index()
df_en_content = df_en_content.rename(columns={'content_text': 'count'})
df_en_content = df_en_content.pivot(index='published_date', columns='sentiment_content', values='count').fillna(0)
df_en_content['date'] = df_en_content.index
df_en_content['date'] = pd.to_datetime(df_en_content['date'])
df_en_content['month'] = df_en_content['date'].dt.month
df_en_content = df_en_content.groupby('month').sum()
col10.write("Sentiment Analysis of Content in en_False.csv Sorted by Date")
col10.line_chart(df_en_content)
st.write("<h5 style='text-align: left;'> In the study of fake news about content analysis within the English language, the analysis reveals a distribution consisting of 460 cases with negative fake news, 10 cases with neutral fake news, and 580 cases exhibiting positive fake news. An examination of the publication dates indicates a notable upward trend in the proliferation of fake news during March 2020. This surge reached its zenith in April 2020, only to subsequently experience a decline commencing in May 2020. There is a very interesting phenomenon in the content analysis, the fake news written in English is rarely neutral, mostly negative or positive, more positive news. </h5>", unsafe_allow_html=True)
st.write("<h5 style='text-align: left;'> In the analysis of content within the English language, the sentiment outcomes of three types showed fake news written is rarely neutral only 1% of total.Positive fake news and negative creatures are equally divided, and positive fake news is more. </h5>", unsafe_allow_html=True)

# Create two columns
col11, col12 = st.columns(2)

# Data visualization for the sentiment analysis result of Content in es_False.csv
df_es_content = df_es.groupby('sentiment_content').count()
col11.write("Sentiment Analysis of Content in es_False.csv")
col11.bar_chart(df_es_content['content_text_translated'])

# Data visualization for the sentiment analysis result of Content in es_False.csv sortby date
df_es_content = df_es.groupby(['published_date', 'sentiment_content']).count().reset_index()
df_es_content = df_es_content.rename(columns={'content_text_translated': 'count'})
df_es_content = df_es_content.pivot(index='published_date', columns='sentiment_content', values='count').fillna(0)
df_es_content['date'] = df_es_content.index
df_es_content['date'] = pd.to_datetime(df_es_content['date'])
df_es_content['month'] = df_es_content['date'].dt.month
df_es_content = df_es_content.groupby('month').sum()
col12.write("Sentiment Analysis of Content in es_False.csv Sorted by Date")
col12.line_chart(df_es_content)
st.write("<h5 style='text-align: left;'> In the study of fake news about content analysis within the Spanish language, the analysis reveals a distribution consisting of 162 cases with negative fake news, 5 cases with neutral fake news, and 140 cases exhibiting positive fake news. An examination of the publication dates indicates a notable upward trend in the proliferation of fake news during March 2020. This surge reached its zenith in April 2020, only to subsequently experience a decline commencing in May 2020. There is a very interesting phenomenon in the content analysis, the fake news written in Spanish is rarely neutral, mostly negative or positive, more positive news. </h5>", unsafe_allow_html=True)
st.write("<h5 style='text-align: left;'> In the analysis of content within the Spanish language, the sentiment outcomes of three types showed fake news written is rarely neutral only 1.6% of total.Positive fake news and negative creatures are equally divided, and negative fake news is more. </h5>", unsafe_allow_html=True)

# Create two columns
col13, col14 = st.columns(2)

# Data visualization for the sentiment analysis result of Content in pt_False.csv
df_pt_content = df_pt.groupby('sentiment_content').count()
col13.write("Sentiment Analysis of Content in pt_False.csv")
col13.bar_chart(df_pt_content['content_text_translated'])

# Data visualization for the sentiment analysis result of Content in pt_False.csv sortby date
df_pt_content = df_pt.groupby(['published_date', 'sentiment_content']).count().reset_index()
df_pt_content = df_pt_content.rename(columns={'content_text_translated': 'count'})
df_pt_content = df_pt_content.pivot(index='published_date', columns='sentiment_content', values='count').fillna(0)
df_pt_content['date'] = df_pt_content.index
df_pt_content['date'] = pd.to_datetime(df_pt_content['date'])
df_pt_content['month'] = df_pt_content['date'].dt.month
df_pt_content = df_pt_content.groupby('month').sum()
col14.write("Sentiment Analysis of Content in pt_False.csv Sorted by Date")
col14.line_chart(df_pt_content)
st.write("<h5 style='text-align: left;'> In the study of fake news about content analysis within the Portuguese language, the analysis reveals a distribution consisting of 96 cases with negative fake news, 2 cases with neutral fake news, and 91 cases exhibiting positive fake news. An examination of the publication dates indicates a notable upward trend in the proliferation of fake news during March 2020. This surge reached its zenith in April 2020, only to subsequently experience a decline commencing in May 2020. There is a very interesting phenomenon in the content analysis, the fake news written in Portuguese is rarely neutral, mostly negative or positive, more positive news. </h5>", unsafe_allow_html=True)
st.write("<h5 style='text-align: left;'> In the analysis of content within the Portuguese language, the sentiment outcomes of three types showed fake news written is rarely neutral only 1% of total.Positive fake news and negative creatures are equally divided, and negative fake news is more. </h5>", unsafe_allow_html=True)

# Create two columns
col15, col16 = st.columns(2)

# Data visualization for the sentiment analysis result of Content in fr_False.csv
df_fr_content = df_fr.groupby('sentiment_content').count()
col15.write("Sentiment Analysis of Content in fr_False.csv")
col15.bar_chart(df_fr_content['content_text_translated'])

# Data visualization for the sentiment analysis result of Content in fr_False.csv sortby date
df_fr_content = df_fr.groupby(['published_date', 'sentiment_content']).count().reset_index()
df_fr_content = df_fr_content.rename(columns={'content_text_translated': 'count'})
df_fr_content = df_fr_content.pivot(index='published_date', columns='sentiment_content', values='count').fillna(0)
df_fr_content['date'] = df_fr_content.index
df_fr_content['date'] = pd.to_datetime(df_fr_content['date'])
df_fr_content['month'] = df_fr_content['date'].dt.month
df_fr_content = df_fr_content.groupby('month').sum()
col16.write("Sentiment Analysis of Content in fr_False.csv Sorted by Date")
col16.line_chart(df_fr_content)
st.write("<h5 style='text-align: left;'> In the study of fake news about content analysis within the French language, the analysis reveals a distribution consisting of 26 cases with negative fake news, 6 cases with neutral fake news, and 52 cases exhibiting positive fake news. An examination of the publication dates indicates a notable upward trend in the proliferation of fake news during March 2020. This surge reached its zenith in April 2020, only to subsequently experience a decline commencing in May 2020. There is a very interesting phenomenon in the content analysis, the fake news written in French is rarely neutral, mostly negative or positive, more positive news. </h5>", unsafe_allow_html=True)
st.write("<h5 style='text-align: left;'> In the analysis of content within the French language, the sentiment outcomes of three types showed fake news written is rarely neutral only 7% of total. Negative fake news is more. </h5>", unsafe_allow_html=True)

st.write("<h3 style='text-align: left;'> Summarize </h3>", unsafe_allow_html=True)
st.write("<h5 style='text-align: left;'> Through the analysis of the Content, four types of language in fake news the articles written in most are negative and positive. The neutral articles are fewer.</h5>", unsafe_allow_html=True)