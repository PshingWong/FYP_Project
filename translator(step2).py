import pandas as pd
import numpy as np
from langdetect import detect
from googletrans import Translator

df_en = pd.read_csv('en_False.csv')
df_es = pd.read_csv('es_False.csv')
df_pt = pd.read_csv('pt_False.csv')
df_fr = pd.read_csv('fr_False.csv')
df_test = pd.read_csv('30_India_False.csv')

translator = Translator()
translated_text_test = []

translated_text_es = []
for i, text in enumerate(df_es['content_text']):
    if text is not None:
        try:
            translated = translator.translate(text, src='es', dest='en').text
            translated_text_es.append(translated)
            print(f"Translated row {i+1} out of {len(df_es)}")
        except:
            translated_text_es.append('')
            print(f"Error translating row {i+1}")
    else:
        translated_text_es.append('')
df_es['content_text_translated'] = translated_text_es
df_es.to_csv('es_False_tran.csv', index=False)

translated_title_es = []
for i, text in enumerate(df_es['source_title']):
    if text is not None:
        try:
            translated = translator.translate(text, src='es', dest='en').text
            translated_title_es.append(translated)
            print(f"Translated row {i+1} out of {len(df_es)}")
        except:
            translated_title_es.append('')
            print(f"Error translating row {i+1}")
    else:
        translated_title_es.append('')
df_es['source_title_translated'] = translated_title_es
df_es.to_csv('es_False_tran.csv', index=False)

translated_text_fr = []
for i, text in enumerate(df_fr['content_text']):
    if text is not None:
        try:
            translated = translator.translate(text, src='fr', dest='en').text
            translated_text_fr.append(translated)
            print(f"Translated row {i+1} out of {len(df_fr)}")
        except:
            translated_text_fr.append('')
            print(f"Error translating row {i+1}")
    else:
        translated_text_fr.append('')
df_fr['content_text_translated'] = translated_text_fr
df_fr.to_csv('fr_False_tran.csv', index=False)

translated_title_fr = []
for i, text in enumerate(df_fr['source_title']):
    if text is not None:
        try:
            translated = translator.translate(text, src='fr', dest='en').text
            translated_title_fr.append(translated)
            print(f"Translated row {i+1} out of {len(df_fr)}")
        except:
            translated_title_fr.append('')
            print(f"Error translating row {i+1}")
    else:
        translated_title_fr.append('')
df_fr['source_title_translated'] = translated_title_fr
df_fr.to_csv('fr_False_tran.csv', index=False)

translated_text_pt = []
for i, text in enumerate(df_pt['content_text']):
    if text is not None:
        try:
            translated = translator.translate(text, src='pt', dest='en').text
            translated_text_pt.append(translated)
            print(f"Translated row {i+1} out of {len(df_pt)}")
        except:
            translated_text_pt.append('')
            print(f"Error translating row {i+1}")
    else:
        translated_text_pt.append('')
df_pt['content_text_translated'] = translated_text_pt
df_pt.to_csv('pt_False_tran.csv', index=False)

translated_title_pt = []
for i, text in enumerate(df_pt['source_title']):
    if text is not None:
        try:
            translated = translator.translate(text, src='pt', dest='en').text
            translated_title_pt.append(translated)
            print(f"Translated row {i+1} out of {len(df_pt)}")
        except:
            translated_title_pt.append('')
            print(f"Error translating row {i+1}")
    else:
        translated_title_pt.append('')
df_pt['source_title_translated'] = translated_title_pt
df_pt.to_csv('pt_False_tran.csv', index=False)






