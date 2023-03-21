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
    page_icon="💭",
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

# Create the Tagcloud page
st.title('Tagcloud Page')
st.write("<h2 style='text-align: left;'> Tagcloud of Title (en)</h2>", unsafe_allow_html=True)
cols5_1 = st.columns(5)
cols5_1[0].image('en_title_Topic_0.png', use_column_width=True)
cols5_1[1].image('en_title_Topic_1.png', use_column_width=True)
cols5_1[2].image('en_title_Topic_2.png', use_column_width=True)
cols5_1[3].image('en_title_Topic_3.png', use_column_width=True)
cols5_1[4].image('en_title_Topic_4.png', use_column_width=True)
cols5_2 = st.columns(5)
cols5_2[0].image('en_title_Topic_5.png', use_column_width=True)
cols5_2[1].image('en_title_Topic_6.png', use_column_width=True)
cols5_2[2].image('en_title_Topic_7.png', use_column_width=True)
cols5_2[3].image('en_title_Topic_8.png', use_column_width=True)
cols5_2[4].image('en_title_Topic_9.png', use_column_width=True)
st.write("<h5 style='text-align: left;'> The tagcloud of title in English is shown above. The tagcloud is divided into 10 topics. The size of the word is related to the frequency of the word. The larger the word, the more frequent the word is. The color of the word is related to the sentiment of the word. The darker the word, the more negative the word is. The lighter the word, the more positive the word is. </h5>", unsafe_allow_html=True)
    
st.write("<h2 style='text-align: left;'> Tagcloud of Content (en)</h2>", unsafe_allow_html=True)
cols6_1 = st.columns(5)
cols6_1[0].image('en_content_Topic_0.png', use_column_width=True)
cols6_1[1].image('en_content_Topic_1.png', use_column_width=True)
cols6_1[2].image('en_content_Topic_2.png', use_column_width=True)
cols6_1[3].image('en_content_Topic_3.png', use_column_width=True)
cols6_1[4].image('en_content_Topic_4.png', use_column_width=True)
cols6_2 = st.columns(5)
cols6_2[0].image('en_content_Topic_5.png', use_column_width=True)
cols6_2[1].image('en_content_Topic_6.png', use_column_width=True)
cols6_2[2].image('en_content_Topic_7.png', use_column_width=True)
cols6_2[3].image('en_content_Topic_8.png', use_column_width=True)
cols6_2[4].image('en_content_Topic_9.png', use_column_width=True)
st.write("<h5 style='text-align: left;'> The tagcloud of content in English is shown above. The tagcloud is divided into 10 topics. The size of the word is related to the frequency of the word. The larger the word, the more frequent the word is. The color of the word is related to the sentiment of the word. The darker the word, the more negative the word is. The lighter the word, the more positive the word is. </h5>", unsafe_allow_html=True)

st.write("<h2 style='text-align: left;'> Tagcloud of Title (es)</h2>", unsafe_allow_html=True)
cols7_1 = st.columns(5)
cols7_1[0].image('es_title_Topic_0.png', use_column_width=True)
cols7_1[1].image('es_title_Topic_1.png', use_column_width=True)
cols7_1[2].image('es_title_Topic_2.png', use_column_width=True)
cols7_1[3].image('es_title_Topic_3.png', use_column_width=True)
cols7_1[4].image('es_title_Topic_4.png', use_column_width=True)
cols7_2 = st.columns(5)
cols7_2[0].image('es_title_Topic_5.png', use_column_width=True)
cols7_2[1].image('es_title_Topic_6.png', use_column_width=True)
cols7_2[2].image('es_title_Topic_7.png', use_column_width=True)
cols7_2[3].image('es_title_Topic_8.png', use_column_width=True)
cols7_2[4].image('es_title_Topic_9.png', use_column_width=True)
st.write("<h5 style='text-align: left;'> The tagcloud of title in Spanish is shown above. The tagcloud is divided into 10 topics. The size of the word is related to the frequency of the word. The larger the word, the more frequent the word is. The color of the word is related to the sentiment of the word. The darker the word, the more negative the word is. The lighter the word, the more positive the word is. </h5>", unsafe_allow_html=True)

st.write("<h2 style='text-align: left;'> Tagcloud of Content (es)</h2>", unsafe_allow_html=True)
cols8_1 = st.columns(5)
cols8_1[0].image('es_content_Topic_0.png', use_column_width=True)
cols8_1[1].image('es_content_Topic_1.png', use_column_width=True)
cols8_1[2].image('es_content_Topic_2.png', use_column_width=True)
cols8_1[3].image('es_content_Topic_3.png', use_column_width=True)
cols8_1[4].image('es_content_Topic_4.png', use_column_width=True)
cols8_2 = st.columns(5)
cols8_2[0].image('es_content_Topic_5.png', use_column_width=True)
cols8_2[1].image('es_content_Topic_6.png', use_column_width=True)
cols8_2[2].image('es_content_Topic_7.png', use_column_width=True)
cols8_2[3].image('es_content_Topic_8.png', use_column_width=True)
cols8_2[4].image('es_content_Topic_9.png', use_column_width=True)
st.write("<h5 style='text-align: left;'> The tagcloud of content in Spanish is shown above. The tagcloud is divided into 10 topics. The size of the word is related to the frequency of the word. The larger the word, the more frequent the word is. The color of the word is related to the sentiment of the word. The darker the word, the more negative the word is. The lighter the word, the more positive the word is. </h5>", unsafe_allow_html=True)

st.write("<h2 style='text-align: left;'> Tagcloud of Title (fr)</h2>", unsafe_allow_html=True)
cols9_1 = st.columns(5)
cols9_1[0].image('fr_title_Topic_0.png', use_column_width=True)
cols9_1[1].image('fr_title_Topic_1.png', use_column_width=True)
cols9_1[2].image('fr_title_Topic_2.png', use_column_width=True)
cols9_1[3].image('fr_title_Topic_3.png', use_column_width=True)
cols9_1[4].image('fr_title_Topic_4.png', use_column_width=True)
cols9_2 = st.columns(5)
cols9_2[0].image('fr_title_Topic_5.png', use_column_width=True)
cols9_2[1].image('fr_title_Topic_6.png', use_column_width=True)
cols9_2[2].image('fr_title_Topic_7.png', use_column_width=True)
cols9_2[3].image('fr_title_Topic_8.png', use_column_width=True)
cols9_2[4].image('fr_title_Topic_9.png', use_column_width=True)
st.write("<h5 style='text-align: left;'> The tagcloud of title in French is shown above. The tagcloud is divided into 10 topics. The size of the word is related to the frequency of the word. The larger the word, the more frequent the word is. The color of the word is related to the sentiment of the word. The darker the word, the more negative the word is. The lighter the word, the more positive the word is. </h5>", unsafe_allow_html=True)

st.write("<h2 style='text-align: left;'> Tagcloud of Content (fr)</h2>", unsafe_allow_html=True)
cols10_1 = st.columns(5)
cols10_1[0].image('fr_content_Topic_0.png', use_column_width=True)
cols10_1[1].image('fr_content_Topic_1.png', use_column_width=True)
cols10_1[2].image('fr_content_Topic_2.png', use_column_width=True)
cols10_1[3].image('fr_content_Topic_3.png', use_column_width=True)
cols10_1[4].image('fr_content_Topic_4.png', use_column_width=True)
cols10_2 = st.columns(5)
cols10_2[0].image('fr_content_Topic_5.png', use_column_width=True)
cols10_2[1].image('fr_content_Topic_6.png', use_column_width=True)
cols10_2[2].image('fr_content_Topic_7.png', use_column_width=True)
cols10_2[3].image('fr_content_Topic_8.png', use_column_width=True)
cols10_2[4].image('fr_content_Topic_9.png', use_column_width=True)
st.write("<h5 style='text-align: left;'> The tagcloud of content in French is shown above. The tagcloud is divided into 10 topics. The size of the word is related to the frequency of the word. The larger the word, the more frequent the word is. The color of the word is related to the sentiment of the word. The darker the word, the more negative the word is. The lighter the word, the more positive the word is. </h5>", unsafe_allow_html=True)
