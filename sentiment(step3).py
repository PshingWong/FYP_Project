import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures

df_en = pd.read_csv('en_False.csv')
df_es = pd.read_csv('es_False_tran.csv')
df_pt = pd.read_csv('pt_False_tran.csv')
df_fr = pd.read_csv('fr_False_tran.csv')
df_test = pd.read_csv('30_India_False.csv')

# Create a new column to store the sentiment analysis of title result in en_False.csv
sid = SentimentIntensityAnalyzer()
df_en['neg_title'] = df_en['title'].apply(lambda x: sid.polarity_scores(x)['neg'])
df_en['neu_title'] = df_en['title'].apply(lambda x: sid.polarity_scores(x)['neu'])
df_en['pos_title'] = df_en['title'].apply(lambda x: sid.polarity_scores(x)['pos'])
df_en['compound_title'] = df_en['title'].apply(lambda x: sid.polarity_scores(x)['compound'])
df_en['sentiment_title'] = df_en['compound_title'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))

# Create a new column to store the sentiment analysis of content result in en_False.csv
df_en['neg_content'] = df_en['content_text'].apply(lambda x: sid.polarity_scores(x)['neg'])
df_en['neu_content'] = df_en['content_text'].apply(lambda x: sid.polarity_scores(x)['neu'])
df_en['pos_content'] = df_en['content_text'].apply(lambda x: sid.polarity_scores(x)['pos'])
df_en['compound_content'] = df_en['content_text'].apply(lambda x: sid.polarity_scores(x)['compound'])
df_en['sentiment_content'] = df_en['compound_content'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))
df_en.to_csv('en_False_tran.csv')

# Create a new column to store the sentiment analysis of title result in es_False.csv
sid = SentimentIntensityAnalyzer()
sid = SentimentIntensityAnalyzer()
df_es['neg_title'] = df_es['source_title_translated'].apply(lambda x: sid.polarity_scores(str(x))['neg'])
df_es['neu_title'] = df_es['source_title_translated'].apply(lambda x: sid.polarity_scores(str(x))['neu'])
df_es['pos_title'] = df_es['source_title_translated'].apply(lambda x: sid.polarity_scores(str(x))['pos'])
df_es['compound_title'] = df_es['source_title_translated'].apply(lambda x: sid.polarity_scores(str(x))['compound'])
df_es['sentiment_title'] = df_es['compound_title'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))

# Create a new column to store the sentiment analysis of content result in es_False.csv
df_es['neg_content'] = df_es['content_text_translated'].apply(lambda x: sid.polarity_scores(str(x))['neg'])
df_es['neu_content'] = df_es['content_text_translated'].apply(lambda x: sid.polarity_scores(str(x))['neu'])
df_es['pos_content'] = df_es['content_text_translated'].apply(lambda x: sid.polarity_scores(str(x))['pos'])
df_es['compound_content'] = df_es['content_text_translated'].apply(lambda x: sid.polarity_scores(str(x))['compound'])
df_es['sentiment_content'] = df_es['compound_content'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))
df_es.to_csv('es_False_tran.csv')

# Create a new column to store the sentiment analysis of title result in pt_False.csv
sid = SentimentIntensityAnalyzer()
df_pt['neg_title'] = df_pt['source_title_translated'].apply(lambda x: sid.polarity_scores(str(x))['neg'])
df_pt['neu_title'] = df_pt['source_title_translated'].apply(lambda x: sid.polarity_scores(str(x))['neu'])
df_pt['pos_title'] = df_pt['source_title_translated'].apply(lambda x: sid.polarity_scores(str(x))['pos'])
df_pt['compound_title'] = df_pt['source_title_translated'].apply(lambda x: sid.polarity_scores(str(x))['compound'])
df_pt['sentiment_title'] = df_pt['compound_title'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))

# Create a new column to store the sentiment analysis of content result in pt_False.csv
df_pt['neg_content'] = df_pt['content_text_translated'].apply(lambda x: sid.polarity_scores(str(x))['neg'])
df_pt['neu_content'] = df_pt['content_text_translated'].apply(lambda x: sid.polarity_scores(str(x))['neu'])
df_pt['pos_content'] = df_pt['content_text_translated'].apply(lambda x: sid.polarity_scores(str(x))['pos'])
df_pt['compound_content'] = df_pt['content_text_translated'].apply(lambda x: sid.polarity_scores(str(x))['compound'])
df_pt['sentiment_content'] = df_pt['compound_content'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))
df_pt.to_csv('pt_False_tran.csv')

# Create a new column to store the sentiment analysis of title result in fr_False.csv
sid = SentimentIntensityAnalyzer()
df_fr['neg_title'] = df_fr['source_title_translated'].apply(lambda x: sid.polarity_scores(str(x))['neg'])
df_fr['neu_title'] = df_fr['source_title_translated'].apply(lambda x: sid.polarity_scores(str(x))['neu'])
df_fr['pos_title'] = df_fr['source_title_translated'].apply(lambda x: sid.polarity_scores(str(x))['pos'])
df_fr['compound_title'] = df_fr['source_title_translated'].apply(lambda x: sid.polarity_scores(str(x))['compound'])
df_fr['sentiment_title'] = df_fr['compound_title'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))

# Create a new column to store the sentiment analysis of content result in fr_False.csv
df_fr['neg_content'] = df_fr['content_text_translated'].apply(lambda x: sid.polarity_scores(str(x))['neg'])
df_fr['neu_content'] = df_fr['content_text_translated'].apply(lambda x: sid.polarity_scores(str(x))['neu'])
df_fr['pos_content'] = df_fr['content_text_translated'].apply(lambda x: sid.polarity_scores(str(x))['pos'])
df_fr['compound_content'] = df_fr['content_text_translated'].apply(lambda x: sid.polarity_scores(str(x))['compound'])
df_fr['sentiment_content'] = df_fr['compound_content'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))
df_fr.to_csv('fr_False_tran.csv')




