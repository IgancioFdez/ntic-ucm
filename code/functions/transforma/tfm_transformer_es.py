# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 09:33:25 2023

@author: ifernandez
"""
#https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/
#https://www.pluralsight.com/guides/building-a-twitter-sentiment-analysis-in-python
#https://huggingface.co/blog/sentiment-analysis-twitter
#https://medium.com/@pragya_paudyal/scraping-tweet-using-twint-and-analyzing-with-nlp-932e01ad5587
#https://www.justintodata.com/twitter-sentiment-analysis-python/
#https://www.analyticsvidhya.com/blog/2021/06/twitter-sentiment-analysis-a-nlp-use-case-for-beginners/
#https://thecleverprogrammer.com/2021/09/13/twitter-sentiment-analysis-using-python/
#https://blog.chapagain.com.np/natural-language-processing-nlp-basic-introduction-to-nltk-python/
#https://blog.chapagain.com.np/python-twitter-sentiment-analysis-on-real-time-tweets-using-textblob/
#https://blog.chapagain.com.np/python-twitter-sentiment-analysis-using-textblob/
#https://blog.chapagain.com.np/python-nltk-sentiment-analysis-on-movie-reviews-natural-language-processing-nlp/
#https://towardsdatascience.com/simple-twitter-analytics-with-twitter-nlp-toolkit-7d7d79bf2535
#https://python.plainenglish.io/nlp-twitter-sentiment-analysis-using-python-ml-4b4a8fc1e2b
#https://blog.chapagain.com.np/python-nltk-twitter-sentiment-analysis-natural-language-processing-nlp/
#https://www.kaggle.com/code/gauravchhabra/nlp-twitter-sentiment-analysis-project
#pip install transformers
#pip install protobuf==3.20.*
#pip install contractions
import pandas as pd
import json
import numpy as np
import re
#from transformers import pipeline
#from nltk.tokenize import TweetTokenizer
import contractions
from google.cloud import bigquery
from google.cloud import language_v1
import os
import fnmatch
import sys

# set key credentials file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'../test/my-test-project-379108-c38ac66b0cd0.json'

# Instantiates a client
client = language_v1.LanguageServiceClient()


# define function that reads from json
def read_tweet_files(file_name): 
	f = open(file_name, "r",encoding="utf-8")
	df=pd.read_json(f)

	#print(df.info)

	return df

def initial_analysis(df):
	#Análisis exploratorio
	#Número de documentos y columnas
	#print("Tenemos un conjunto de {} documentos".format(len(df)))
	#print("El dataframe tiene {} columnas".format(df.shape[1]))

	#Número de documentos duplicados
	#print("Existen {} tweets duplicados".format(np.sum(df.duplicated(subset=["Tweet"]))))


	#Comprobaramos que no hayan quedado Nulls en ningunas de las dos columnas del dataset.
	#print("Hay {} valores vacíos en los tweets ".format(np.sum(df.isnull())[3]))

	#Distribución de la longitud de los tweets en caracteres
	df["Char_Len"] = df["Tweet"].apply(lambda x: len(x))
	
	#print(df.columns)
	
	return df

#Normalización
# Eliminar saltos de línea
def eliminar_salto_linea(text):
	 return  re.sub('\n','',text) 

#Eliminar espacios extra
def eliminar_espacios(text): 
    return  re.sub(r'\s+', ' ', text, flags=re.I)

#Eliminar single caracteres
def single_char(text): 
    return  re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

# Eliminar url
def eliminar_url(text): 
    return  re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

# Eliminar url
def correct_url(text): 
    return  text.replace("\\", '')

# Pasar a minúsculas
def texto_to_lower(text):
  return text.lower()

# Pasar a minúsculas
def remove_mentions_hastags(text):
  return re.sub(r'\@\w+|\#\w+','', text)

# # Reemplazar contracciones y slang en inglés usando la librería "contractions" https://github.com/kootenpv/contractions
# def replace_contraction(text):
#     expanded_words = []
#     # Divide el texto
#     for t in text.split():
#         # Aplica la función fix en cada sección o token del texto buscando contracciones y slang
#         expanded_words.append(contractions.fix(t, slang = True))
#     expanded_text = ' '.join(expanded_words) 
#     return expanded_text

# # Eliminar los emojis de un texto. Esto es útil porque una vez extraido los emojis
# # puede interesarnos tener un texto sin presencia de emojis para mejor análisis.
def clean_emoji(text):
    # Poner todos los comandos de http://www.unicode.org/Public/emoji/1.0/emoji-data.txt
    emoji_pattern = re.compile("["
        u"\U0001F300-\U0001F6FF"  # symbols & pictographs
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u'\u2600-\u26FF\u2700-\u27BF'
        u'\u2934' u'\u2935' u'\u2B05' u'\u2B06' u'\u2B07' u'\u2B1B' u'\u2B1C' 
        u'\u2B50' u'\u2B55' u'\u3030' u'\u303D' u'\u3297' u'\u3299' u'\u00A9'
        u'\u00AE' u'\u203C' u'\u2049' u'\u2122' u'\u2139' u'\u2194-\u2199' 
        u'\u21A9' u'\u21AA' u'\u231A' u'\u231B' u'\u2328' u'\u23CF'
        u'\u23E9-\u23F3' u'\u23F8' u'\u23F9' u'\u23FA' u'\u24C2' u'\u25AA'
        u'\u25AB' u'\u25B6' u'\u25C0' u'\u25FB' u'\u25FD' u'\u25FC' u'\u25FE'
        ']+', flags=re.UNICODE)
    string2 = re.sub(emoji_pattern,r' ',text)
    return string2

# # Quitar stop words de una lista de tokens
# def quitar_stopwords(tokens):
#     stop_words = set(stopwords.words('english')) 
#     filtered_sentence = [w for w in tokens if not w in stop_words]
#     return filtered_sentence

# # Eliminar signos de puntuación de una lista de tokens
# # (nos quedamos sólo lo alfanumérico en este caso)
# def quitar_puntuacion(tokens):
#     words=[word for word in tokens if word.isalnum()]
#     return words

def get_sentiment(text,file):
	document = language_v1.Document(
	    content=text, type_=language_v1.Document.Type.PLAIN_TEXT
	)

	try:
		# Detects the sentiment of the text
		sentiment = client.analyze_sentiment(
		    request={"document": document}
		).document_sentiment
	except Exception:
		print('[SENTIMENT] : ERROR en el fichero  ::>' + file)
		return 0.0

	#print("Text: {}".format(text))
	#print("Sentiment: {}, {}".format(sentiment.score, sentiment.magnitude))
	return round(sentiment.score, 4)

def export_items_to_bigquery(id,date,text,lang,likes,replies,retweets,hashtags,link,country,sentiment,processed):
    # Instantiates a client
    bigquery_client = bigquery.Client()

    # Prepares a reference to the dataset
    dataset_ref = bigquery_client.dataset('tweets_wc_ntic_master')

    table_ref = dataset_ref.table('tweets_test')
    table = bigquery_client.get_table(table_ref)  # API call

    rows_to_insert = [(id,date,text,lang,likes,replies,retweets,hashtags,link,country,sentiment,processed)]
    errors = bigquery_client.insert_rows(table, rows_to_insert)  # API request
    assert errors == []

def app():
	print(sys.argv[1])
	filesOfDirectory = os.listdir(sys.argv[1])
	pattern = "*_es_*"
	for file in filesOfDirectory:
		if fnmatch.fnmatch(file, pattern):
 			#print(file)
			df=read_tweet_files(sys.argv[1]+"/"+file)
			
			df=initial_analysis(df)
			df.columns
			
			df["Tweet_Processed"] = df["Tweet"].apply(lambda x: eliminar_salto_linea(x))
			df["Tweet_Processed"] = df["Tweet_Processed"].apply(lambda x: eliminar_espacios(x))
			df["Tweet_Processed"] = df["Tweet"].apply(lambda x: single_char(x))
			df["Tweet_Processed"] = df["Tweet_Processed"].apply(lambda x: eliminar_url(x))
			df["Tweet_Processed"] = df["Tweet_Processed"].apply(lambda x: remove_mentions_hastags(x))
			df["Tweet_Processed"] = df["Tweet_Processed"].apply(lambda x: texto_to_lower(x))
			df["Tweet_Processed"] = df["Tweet_Processed"].apply(lambda x: clean_emoji(x))
			df["Tweet_Sentiment"] = df["Tweet_Processed"].apply(lambda x: get_sentiment(x,file))
			df["Link"] = df["Link"].apply(lambda x: correct_url(str(x)))
			print("[SENTIMENT RESULT] === Fichero  :  " + file + "  con total registros  :  " + str(df.shape[0]))
			
		
# 			for ind in df.index:
#  				 export_items_to_bigquery(str(df['Tweet_Id'][ind]),str(df['Date'][ind]),str(df['Tweet'][ind]),str(df['Language'][ind]),int(df['Likes_Count'][ind]),int(df['Replies_Count'][ind]),int(df['Retweets_Count'][ind]),df['Hashtags'][ind],str(df['Link'][ind]),str(df['Country'][ind]),float(df['Tweet_Sentiment'][ind]),str(df['Tweet_Processed'][ind]))

if __name__ == '__main__':
	app()