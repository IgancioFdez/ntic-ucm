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
from transformers import pipeline
from nltk.tokenize import TweetTokenizer
import contractions


# define function that reads from json
def read_tweet_files(file_name): 
# 	print(file_name)
# 	with open(file_name, "r",encoding='utf-8') as data:
# 		  parsed_json = json.load(data)

# 	print(parsed_json)

	df = pd.read_json(file_name)
	print(df.head())
	print(df.columns)

	return df

def initial_analysis(df):
	#Análisis exploratorio
	#Número de documentos y columnas
	print("Tenemos un conjunto de {} documentos".format(len(df)))
	print("El dataframe tiene {} columnas".format(df.shape[1]))

	#Número de documentos duplicados
	print("Existen {} tweets duplicados".format(np.sum(df.duplicated(subset=["Tweet"]))))


	#Comprobaramos que no hayan quedado Nulls en ningunas de las dos columnas del dataset.
	print("Hay {} valores vacíos en los tweets ".format(np.sum(df.isnull())[3]))

	#Distribución de la longitud de los tweets en caracteres
	df["Char_Len"] = df["Tweet"].apply(lambda x: len(x))
	
	print(df.columns)
	
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

# Pasar a minúsculas
def texto_to_lower(text):
  return text.lower()

# Pasar a minúsculas
def remove_mentions_hastags(text):
  return re.sub(r'\@\w+|\#\w+','', text)

# Preparar diccionario de emojis
def load_emoji_sentiment(path):
  # Cargamos el csv de emoji_sentiment
  emoji_sent_df = pd.read_csv(path,sep=",")
  # Calculamos los scores dividiendo el número de emojis negativos y entre el total
  emoji_sent_df["Negative"] = emoji_sent_df["Negative"]/emoji_sent_df["Occurrences"]
  emoji_sent_df["Neutral"] = emoji_sent_df["Neutral"]/emoji_sent_df["Occurrences"]
  emoji_sent_df["Positive"] = emoji_sent_df["Positive"]/emoji_sent_df["Occurrences"]
  # Transformamos a dict
  emoji_sent_df = emoji_sent_df.set_index('Emoji')
  emoji_dict = emoji_sent_df.to_dict(orient="index")
  return emoji_dict

# Reemplazar contracciones y slang en inglés usando la librería "contractions" https://github.com/kootenpv/contractions
def replace_contraction(text):
    expanded_words = []
    # Divide el texto
    for t in text.split():
        # Aplica la función fix en cada sección o token del texto buscando contracciones y slang
        expanded_words.append(contractions.fix(t, slang = True))
    expanded_text = ' '.join(expanded_words) 
    return expanded_text

# Función para extraer emojis del texto en formato lista
def extract_emojis(text):
  extract = Extractor()
  emojis = extract.count_emoji(text, check_first=False)
  emojis_list = [key for key, _ in emojis.most_common()]
  return emojis_list

# Calcula el sentimiento de los emojis de una lista utilizando el diccionario
# de emoji sentiment score generado previamente con la función load_emoji_sentiment()
# Se puede extraer el valor de positividad de los emojis con la option "positive"
# Se puede extraer el valor de neutralidad de los emojis con la option "neutral""  
# Se puede extraer el valor de e negatividad de los emojis con la option "negative""  

def get_emoji_sentiment(lista, option = "positive"):
  output = 0
  for emoji in lista:
    try:
      if option == "positive":
        output = output + emoji_sent_dict[emoji]["Positive"]
      elif option =="negative":
        output = output + emoji_sent_dict[emoji]["Negative"]
      elif option =="neutral":
        output = output + emoji_sent_dict[emoji]["Neutral"]
    except Exception as e: 
      continue
  return output

# Eliminar los emojis de un texto. Esto es útil porque una vez extraido los emojis
# puede interesarnos tener un texto sin presencia de emojis para mejor análisis.
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

# Tokenizador
def tokenize(texto):
    tweet_tokenizer = TweetTokenizer()
    tokens_list = tweet_tokenizer.tokenize(texto)
    return tokens_list

# Quitar stop words de una lista de tokens
def quitar_stopwords(tokens):
    stop_words = set(stopwords.words('english')) 
    filtered_sentence = [w for w in tokens if not w in stop_words]
    return filtered_sentence

# Eliminar signos de puntuación de una lista de tokens
# (nos quedamos sólo lo alfanumérico en este caso)
def quitar_puntuacion(tokens):
    words=[word for word in tokens if word.isalnum()]
    return words


def app():
	
	df=read_tweet_files('./tweets.json')
	
	df=initial_analysis(df)
	
	df["Tweet_Processed"] = df["Tweet"].apply(lambda x: eliminar_salto_linea(x))
	df["Tweet_Processed"] = df["Tweet_Processed"].apply(lambda x: eliminar_espacios(x))
	df["Tweet_Processed"] = df["Tweet"].apply(lambda x: single_char(x))
	df["Tweet_Processed"] = df["Tweet_Processed"].apply(lambda x: eliminar_url(x))
	df["Tweet_Processed"] = df["Tweet_Processed"].apply(lambda x: remove_mentions_hastags(x))
	df["Tweet_Processed"] = df["Tweet_Processed"].apply(lambda x: texto_to_lower(x))
# 	df["Tweet_Processed"] = df["Tweet_Processed"].apply(lambda x: tokenize(x))
# 	df["Tweet_Processed"] = df["Tweet_Processed"].apply(lambda x: quitar_stopwords(x))
# 	df["Tweet_Processed"] = df["Tweet_Processed"].apply(lambda x: quitar_puntuacion(x))
# 	df["Tweet_Processed"] = df["Tweet_Processed"].apply(lambda x: lematizar(x))

	#model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
	#sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
	#sentiment_task("T'estimo!")
	#df["Sentiment_0"] = df["Tweet_Processed"].apply(lambda x: sentiment_task(x))

	print(df["Tweet_Processed"])


if __name__ == '__main__':
	app()