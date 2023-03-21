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
import string

df = pd.read_csv('FakeCovid_July2020.csv')

# cleanning unnessary columns
df = df.drop(columns = ['country1', 'country2', 'country3', 'country4', 'category', 'ref_source']) 
df = df.dropna()
df = df.reset_index(drop=True)
df = df.drop_duplicates()
df.fillna("NA", inplace=True)

# data type conversion
df['class'] = df['class'].astype(str)
df['lang'] = df['lang'].astype(str)
df['country'] = df['country'].astype(str)
df['content_text'] = df['content_text'].astype(str)
df['title'] = df['title'].astype(str)
df['published_date'] = df['published_date'].astype(str)

# count the number of fake news in each country
df_country = df.groupby(['country']).size().reset_index(name='counts')
df_country = df_country.sort_values(by=['counts'], ascending=False)
df_country = df_country[df_country['counts'] > 50]
plt.figure(figsize=(10, 5))
plt.bar(df_country['country'], df_country['counts'])
plt.xticks(rotation=90) # rotate x-axis labels
plt.xlabel('Country')
plt.ylabel('Count')
plt.title('Number of Fake News by Country')
plt.savefig('country.png', bbox_inches='tight') # save figure with tight bounding box
plt.show()
# India and Us have the most fake news

#count the number of fake news in each language
df_lang = df.groupby(['lang']).size().reset_index(name='counts')
df_lang = df_lang.sort_values(by=['counts'], ascending=False)
df_lang = df_lang[df_lang['counts'] > 50]
plt.figure(figsize=(10, 5))
plt.bar(df_lang['lang'], df_lang['counts'])
plt.xticks(rotation=90) # rotate x-axis labels
plt.xlabel('Language')
plt.ylabel('Count')
plt.title('Number of Fake News by Language')
plt.savefig('lang.png', bbox_inches='tight') # save figure with tight bounding box
# plt.show()
# en and es have the most fake news

# create a new dataframe with only en fake news
df_en = df[(df['class'] =='FALSE') & (df['lang'] == 'en')]
df_en.to_csv('en_False.csv')
# print(df)

# create a new dataframe with only es fake news
df_es = df[(df['class'] =='FALSE') & (df['lang'] == 'es')]
df_es.to_csv('es_False.csv')
# print(df)

# create a new dataframe with only pt fake news
df_pt = df[(df['class'] =='FALSE') & (df['lang'] == 'pt')]
df_pt.to_csv('pt_False.csv')
# print(df)

# create a new dataframe with only fr fake news
df_fr = df[(df['class'] =='FALSE') & (df['lang'] == 'fr')]
df_fr.to_csv('fr_False.csv')
# print(df)

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

# Remove all emoji from DataFrame
df_en['content_text'] = df_en['content_text'].apply(lambda x: emoji_pattern.sub(r'', x))
df_es['content_text'] = df_es['content_text'].apply(lambda x: emoji_pattern.sub(r'', x))
df_pt['content_text'] = df_pt['content_text'].apply(lambda x: emoji_pattern.sub(r'', x))
df_fr['content_text'] = df_fr['content_text'].apply(lambda x: emoji_pattern.sub(r'', x))

df_en['source_title'] = df_en['source_title'].apply(lambda x: emoji_pattern.sub(r'', x))
df_es['source_title'] = df_es['source_title'].apply(lambda x: emoji_pattern.sub(r'', x))
df_pt['source_title'] = df_pt['source_title'].apply(lambda x: emoji_pattern.sub(r'', x))
df_fr['source_title'] = df_fr['source_title'].apply(lambda x: emoji_pattern.sub(r'', x))

# Define a function to remove punctuation
def remove_punctuation(text):
    # Define a translation table to remove all punctuation
    translator = str.maketrans('', '', string.punctuation)
    # Remove punctuation from the text
    text = text.translate(translator)
    return text

# Apply the remove_punctuation function to the text column of DataFrame
df_en['content_text'] = df_en['content_text'].apply(remove_punctuation)
df_es['content_text'] = df_es['content_text'].apply(remove_punctuation)
df_pt['content_text'] = df_pt['content_text'].apply(remove_punctuation)
df_fr['content_text'] = df_fr['content_text'].apply(remove_punctuation)

df_en['source_title'] = df_en['source_title'].apply(remove_punctuation)
df_es['source_title'] = df_es['source_title'].apply(remove_punctuation)
df_pt['source_title'] = df_pt['source_title'].apply(remove_punctuation)
df_fr['source_title'] = df_fr['source_title'].apply(remove_punctuation)

# Write DataFrame back to CSV file
df_en.to_csv('en_False.csv', index=False)
df_es.to_csv('es_False.csv', index=False)
df_pt.to_csv('pt_False.csv', index=False)
df_fr.to_csv('fr_False.csv', index=False)





