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
import plotly.express as px


df_original = pd.read_csv('FakeCovid_July2020.csv')
df_en = pd.read_csv('en_False_tran.csv')
df_es = pd.read_csv('es_False_tran.csv')
df_pt = pd.read_csv('pt_False_tran.csv')
df_fr = pd.read_csv('fr_False_tran.csv')

st.set_page_config(
    page_icon="🧊",
    layout="wide",
)


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)


# Add custom CSS
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

# Create the home page
st.title('Home Page')
st.markdown("<h1 style='text-align: center;'>Topic: Interactive visualization of author sentiments in health-related news</h1>", unsafe_allow_html=True)
st.write("<h3 style='text-align: center;'>Student Name: Wong Pui Shing</h3>", unsafe_allow_html=True)
st.write("<h3 style='text-align: center;'>Student ID:21204438</h3>", unsafe_allow_html=True)
image = Image.open('image_1.jpeg')
st.image(image, use_column_width=True)
st.write("<h2 style='text-align: left;'> Introduction</h2>", unsafe_allow_html=True)
st.write("<h5 style='text-align: left;'> The outbreak of the COVID-19 pandemic has resulted in an unprecedented global health crisis, accompanied by a rapid proliferation of misinformation. This study examines the sentiments and aspects present in health-related news articles discussing COVID-19 misinformation during the year 2020. By conducting a thorough analysis of these articles, the project aims to find fake news in different country interesting analysis. </h5>", unsafe_allow_html=True)

# Create a bar chart in Streamlit

# cleanning unnessary columns
df_original = df_original.drop(columns = ['country1', 'country2', 'country3', 'country4', 'category', 'ref_source']) 
df_original = df_original.dropna()
df_original = df_original.reset_index(drop=True)
df_original = df_original.drop_duplicates()
df_original.fillna("NA", inplace=True)

# data type conversion
df_original['class'] = df_original['class'].astype(str)
df_original['lang'] = df_original['lang'].astype(str)
df_original['country'] = df_original['country'].astype(str)
df_original['content_text'] = df_original['content_text'].astype(str)
df_original['title'] = df_original['title'].astype(str)

df_lang = df_original.groupby(['lang']).size().reset_index(name='counts')
df_lang = df_lang[df_lang['counts'] > 50]
df_lang_sorted = df_lang.sort_values(by='counts', ascending=False) 

fig = px.bar(df_lang_sorted, x='lang', y='counts', text='counts')

# Update layout properties
fig.update_layout(
    title="Number of Fake News by Language",
    xaxis_title="Language",
    yaxis_title="Count",
    showlegend=False,
    font=dict(size=14),
    autosize=False,
)
col1, col2 = st.columns(2)

# Display the bar chart in Streamlit
col1.plotly_chart(fig)

# Sort DataFrame by 'counts' column in descending order (max to min)
df_lang_sorted = df_lang.sort_values(by='counts', ascending=False)

st.write("Sunburst Chart: Number of Fake News by Language")

# Create a Plotly sunburst chart
fig = px.sunburst(df_lang_sorted, path=['lang'], values='counts', title="Number of Fake News by Language")

# Update layout properties
fig.update_layout(
    font=dict(size=14),
    autosize=True,
)

# Display the sunburst chart in Streamlit
col2.plotly_chart(fig)

st.write("<h5 style='text-align: left;'> The dataset collected in from 92 fact-checking websites, encompassing fact-checked articles published between April 1, 2020, and July 1, 2020. These articles were subsequently sorted based on their respective languages.Data analysis showed that articles written in English contained the highest number of fake news, followed by those in Spanish, French, and Portuguese.</h5>", unsafe_allow_html=True)
st.write("<h5 style='text-align: left;'>Next, it will be split into four different data in these four languages and analyzed through the title and content of articles to discover some interesting insights about health fake news.</h5>", unsafe_allow_html=True)