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

df_en = pd.read_csv('es_False_tran.csv')
df_es = pd.read_csv('es_False_tran.csv')
df_pt = pd.read_csv('pt_False_tran.csv')
df_fr = pd.read_csv('fr_False_tran.csv')
df_test = pd.read_csv('30_India_False.csv')

# Data visualization for the sentiment analysis result of title in en_False.csv
df_en_title = df_en.groupby('sentiment_title').count()
plt.title('Sentiment Analysis of Title in en_False.csv')
plt.pie(df_en_title['title'], labels = df_en_title.index, autopct = '%1.2f%%')
plt.savefig('Sentiment Analysis of Title in en_False')
plt.show()

# Data visualization for the sentiment analysis result of title in en_False.csv sortby date
df_en_title = df_en.groupby(['published_date', 'sentiment_title']).count().reset_index()
df_en_title = df_en_title.rename(columns = {'title': 'count'})
df_en_title = df_en_title.pivot(index = 'published_date', columns = 'sentiment_title', values = 'count').fillna(0)
df_en_title['date'] = df_en_title.index
df_en_title['date'] = pd.to_datetime(df_en_title['date'])
df_en_title['month'] = df_en_title['date'].dt.month
df_en_title = df_en_title.groupby('month').sum()
df_en_title.plot(kind = 'bar', figsize = (20, 10), title = 'Sentiment Analysis of Title in en_False.csv')
plt.savefig('Sentiment Analysis of Title in en_False sortby date of month')
plt.show()

# Data visualization for the sentiment analysis result of content in en_False.csv
df_en_content = df_en.groupby('sentiment_content').count()
plt.title('Sentiment Analysis of Content in en_False.csv')
plt.pie(df_en_content['content_text'], labels = df_en_content.index, autopct = '%1.2f%%')
plt.savefig('Sentiment Analysis of Content in en_False')
plt.show()

# Data visualization for the sentiment analysis result of content in en_False.csv sortby date
df_en_content = df_en.groupby(['published_date', 'sentiment_content']).count().reset_index()
df_en_content = df_en_content.rename(columns = {'content_text': 'count'})
df_en_content = df_en_content.pivot(index = 'published_date', columns = 'sentiment_content', values = 'count').fillna(0)
df_en_content['date'] = df_en_content.index
df_en_content['date'] = pd.to_datetime(df_en_content['date'])
df_en_content['month'] = df_en_content['date'].dt.month
df_en_content = df_en_content.groupby('month').sum()
df_en_content.plot(kind = 'bar', figsize = (10, 5), title = 'Sentiment Analysis of Content in en_False.csv')
plt.savefig('Sentiment Analysis of Content in en_False sortby date of month')
plt.show()

# Data visualization for the sentiment analysis result of title in es_False.csv
df_es_title = df_es.groupby('sentiment_title').count()
plt.title('Sentiment Analysis of Title in es_False.csv')
plt.pie(df_es_title['title'], labels = df_es_title.index, autopct = '%1.2f%%')
plt.savefig('Sentiment Analysis of Title in es_False')
plt.show()

# Data visualization for the sentiment analysis result of title in es_False.csv sortby date
df_es_title = df_es.groupby(['published_date', 'sentiment_title']).count().reset_index()
df_es_title = df_es_title.rename(columns = {'title': 'count'})
df_es_title = df_es_title.pivot(index = 'published_date', columns = 'sentiment_title', values = 'count').fillna(0)
df_es_title['date'] = df_es_title.index
df_es_title['date'] = pd.to_datetime(df_es_title['date'])
df_es_title['month'] = df_es_title['date'].dt.month
df_es_title = df_es_title.groupby('month').sum()
df_es_title.plot(kind = 'bar', figsize = (10, 5), title = 'Sentiment Analysis of Title in es_False.csv')
plt.savefig('Sentiment Analysis of Title in es_False sortby date of month')
plt.show()

# Data visualization for the sentiment analysis result of content in es_False.csv
df_es_content = df_es.groupby('sentiment_content').count()
plt.title('Sentiment Analysis of Content in es_False.csv')
plt.pie(df_es_content['content_text'], labels = df_es_content.index, autopct = '%1.2f%%')
plt.savefig('Sentiment Analysis of Content in es_False')
plt.show()

# Data visualization for the sentiment analysis result of content in es_False.csv sortby date
df_es_content = df_es.groupby(['published_date', 'sentiment_content']).count().reset_index()
df_es_content = df_es_content.rename(columns = {'content_text': 'count'})
df_es_content = df_es_content.pivot(index = 'published_date', columns = 'sentiment_content', values = 'count').fillna(0)
df_es_content['date'] = df_es_content.index
df_es_content['date'] = pd.to_datetime(df_es_content['date'])
df_es_content['month'] = df_es_content['date'].dt.month
df_es_content = df_es_content.groupby('month').sum()
df_es_content.plot(kind = 'bar', figsize = (10, 5), title = 'Sentiment Analysis of Content in es_False.csv')
plt.savefig('Sentiment Analysis of Content in es_False sortby date of month')
plt.show()

# Data visualization for the sentiment analysis result of title in fr_False.csv
df_fr_title = df_fr.groupby('sentiment_title').count()
plt.title('Sentiment Analysis of Title in fr_False.csv')
plt.pie(df_fr_title['title'], labels = df_fr_title.index, autopct = '%1.2f%%')
plt.savefig('Sentiment Analysis of Title in fr_False')
plt.show()

# Data visualization for the sentiment analysis result of title in fr_False.csv sortby date
df_fr_title = df_fr.groupby(['published_date', 'sentiment_title']).count().reset_index()
df_fr_title = df_fr_title.rename(columns = {'title': 'count'})
df_fr_title = df_fr_title.pivot(index = 'published_date', columns = 'sentiment_title', values = 'count').fillna(0)
df_fr_title['date'] = df_fr_title.index
df_fr_title['date'] = pd.to_datetime(df_fr_title['date'])
df_fr_title['month'] = df_fr_title['date'].dt.month
df_fr_title = df_fr_title.groupby('month').sum()
df_fr_title.plot(kind = 'bar', figsize = (10, 5), title = 'Sentiment Analysis of Title in fr_False.csv')
plt.savefig('Sentiment Analysis of Title in fr_False sortby date of month')
plt.show()

# Data visualization for the sentiment analysis result of content in fr_False.csv
df_fr_content = df_fr.groupby('sentiment_content').count()
plt.title('Sentiment Analysis of Content in fr_False.csv')
plt.pie(df_fr_content['content_text'], labels = df_fr_content.index, autopct = '%1.2f%%')
plt.savefig('Sentiment Analysis of Content in fr_False')
plt.show()

# Data visualization for the sentiment analysis result of content in fr_False.csv sortby date
df_fr_content = df_fr.groupby(['published_date', 'sentiment_content']).count().reset_index()
df_fr_content = df_fr_content.rename(columns = {'content_text': 'count'})
df_fr_content = df_fr_content.pivot(index = 'published_date', columns = 'sentiment_content', values = 'count').fillna(0)
df_fr_content['date'] = df_fr_content.index
df_fr_content['date'] = pd.to_datetime(df_fr_content['date'])
df_fr_content['month'] = df_fr_content['date'].dt.month
df_fr_content = df_fr_content.groupby('month').sum()
df_fr_content.plot(kind = 'bar', figsize = (10, 5), title = 'Sentiment Analysis of Content in fr_False.csv')
plt.savefig('Sentiment Analysis of Content in fr_False sortby date of month')
plt.show()

# Data visualization for the sentiment analysis result of title in pt_False.csv
df_pt_title = df_pt.groupby('sentiment_title').count()
plt.title('Sentiment Analysis of Title in pt_False.csv')
plt.pie(df_pt_title['title'], labels = df_pt_title.index, autopct = '%1.2f%%')
plt.savefig('Sentiment Analysis of Title in pt_False')
plt.show()

# Data visualization for the sentiment analysis result of title in pt_False.csv sortby date
df_pt_title = df_pt.groupby(['published_date', 'sentiment_title']).count().reset_index()
df_pt_title = df_pt_title.rename(columns = {'title': 'count'})
df_pt_title = df_pt_title.pivot(index = 'published_date', columns = 'sentiment_title', values = 'count').fillna(0)
df_pt_title['date'] = df_pt_title.index
df_pt_title['date'] = pd.to_datetime(df_pt_title['date'])
df_pt_title['month'] = df_pt_title['date'].dt.month
df_pt_title = df_pt_title.groupby('month').sum()
df_pt_title.plot(kind = 'bar', figsize = (10, 5), title = 'Sentiment Analysis of Title in pt_False.csv')
plt.savefig('Sentiment Analysis of Title in pt_False sortby date of month')
plt.show()

# Data visualization for the sentiment analysis result of content in pt_False.csv
df_pt_content = df_pt.groupby('sentiment_content').count()
plt.title('Sentiment Analysis of Content in pt_False.csv')
plt.pie(df_pt_content['content_text'], labels = df_pt_content.index, autopct = '%1.2f%%')
plt.savefig('Sentiment Analysis of Content in pt_False')
plt.show()

# Data visualization for the sentiment analysis result of content in pt_False.csv sortby date
df_pt_content = df_pt.groupby(['published_date', 'sentiment_content']).count().reset_index()
df_pt_content = df_pt_content.rename(columns = {'content_text': 'count'})
df_pt_content = df_pt_content.pivot(index = 'published_date', columns = 'sentiment_content', values = 'count').fillna(0)
df_pt_content['date'] = df_pt_content.index
df_pt_content['date'] = pd.to_datetime(df_pt_content['date'])
df_pt_content['month'] = df_pt_content['date'].dt.month 
df_pt_content = df_pt_content.groupby('month').sum()
df_pt_content.plot(kind = 'bar', figsize = (10, 5), title = 'Sentiment Analysis of Content in pt_False.csv')
plt.savefig('Sentiment Analysis of Content in pt_False sortby date of month')
plt.show()

