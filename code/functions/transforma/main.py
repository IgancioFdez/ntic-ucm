# -*- coding: utf-8 -*-

import pandas as pd
import json
import re
from google.cloud import bigquery
from google.cloud import language_v1
from google.cloud import storage
import os
import fnmatch
import base64
import functions_framework


bucket_name = 'my-test-project-bucket-ntic'

# Instantiates a client
client = language_v1.LanguageServiceClient()

# Function to read from json
def read_tweet_files(file): 

	df = pd.DataFrame(file)
	#print(df.info)

	return df

# Function to explore df
def initial_analysis(df):
	#Análisis exploratorio
	#Número de documentos y columnas
	print("Tenemos un conjunto de {} documentos".format(len(df)))
	#print("El dataframe tiene {} columnas".format(df.shape[1]))

	#Distribución de la longitud de los tweets en caracteres
	df["Char_Len"] = df["Tweet"].apply(lambda x: len(x))
	
	#print(df.columns)
	
	return df

# Function to remove new lines character
def remove_new_line(text):
	 return  re.sub('\n','',text) 

# Function to remove extra spaces
def remove_spaces(text): 
    return  re.sub(r'\s+', ' ', text, flags=re.I)

# Function to remove single characters
def remove_single_char(text): 
    return  re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

# Function to remove urls
def remove_url(text): 
    return  re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

# Function to fix tweet link
def correct_url(text): 
    return  text.replace("\\", '')

# Function to convert lowercase
def texto_to_lower(text):
  return text.lower()

# Function to remove hashtags
def remove_mentions_hastags(text):
  return re.sub(r'\@\w+|\#\w+','', text)

# Function to remove emoji
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

# Function to list files from bucket
def list_cs_files(bucket_name,path_to_files):
	storage_client = storage.Client()
	#logging.info('Meensaje con ID: ' + storage_client)
	
	file_list = storage_client.list_blobs(bucket_name, prefix=path_to_files)
	file_list = [file.name for file in file_list]
	#print(file_list)
	return file_list
	
# Function to get file from bucket
def read_cs_file(bucket_name,file_name): 
	storage_client = storage.Client()
	#logging.info('Meensaje con ID: ' + storage_client)
	
	# get bucket with name
	bucket = storage_client.get_bucket(bucket_name)
	# get bucket data as blob
	blob = bucket.get_blob(file_name)

	file=blob.download_as_string()
	#print(file) 
	#downloaded_blob = blob.download_as_text()
	file = file.decode("utf-8")

	#print(json.loads(file))
	tweets=json.loads(file)
	#print(len(tweets))

	return tweets
	
# Function to get sentiment from text
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

# Function to insert in bigquery
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

# Triggered from a message on a Cloud Pub/Sub topic.
@functions_framework.cloud_event
def app(cloud_event):
	# Print out the data from Pub/Sub, to prove that it worked
	data=base64.b64decode(cloud_event.data["message"]["data"])
	data=data.decode("utf-8")
	
	path_to_files='data/json/tweets/unified/' + data;
	
	files=list_cs_files(bucket_name,path_to_files)
	pattern = "*_en_*"
	for file in files:
		
		if fnmatch.fnmatch(file, pattern):
			
			f=read_cs_file(bucket_name,file)

			df=read_tweet_files(f)
			
			df=initial_analysis(df)
			#df.columns
			
			df["Tweet_Processed"] = df["Tweet"].apply(lambda x: remove_new_line(x))
			df["Tweet_Processed"] = df["Tweet_Processed"].apply(lambda x: remove_spaces(x))
			df["Tweet_Processed"] = df["Tweet_Processed"].apply(lambda x: remove_single_char(x))
			df["Tweet_Processed"] = df["Tweet_Processed"].apply(lambda x: remove_url(x))
			df["Tweet_Processed"] = df["Tweet_Processed"].apply(lambda x: remove_mentions_hastags(x))
			df["Tweet_Processed"] = df["Tweet_Processed"].apply(lambda x: texto_to_lower(x))
			df["Tweet_Processed"] = df["Tweet_Processed"].apply(lambda x: clean_emoji(x))
			df["Tweet_Sentiment"] = df["Tweet_Processed"].apply(lambda x: get_sentiment(x,file))
			df["Link"] = df["Link"].apply(lambda x: correct_url(str(x)))
			#print("[SENTIMENT RESULT] === Fichero  :  " + file + "  con total registros  :  " + str(df.shape[0]))
			
			for ind in df.index:
 				 export_items_to_bigquery(str(df['Tweet_Id'][ind]),str(df['Date'][ind]),str(df['Tweet'][ind]),str(df['Language'][ind]),int(df['Likes_Count'][ind]),int(df['Replies_Count'][ind]),int(df['Retweets_Count'][ind]),df['Hashtags'][ind],str(df['Link'][ind]),str(df['Country'][ind]),float(df['Tweet_Sentiment'][ind]),str(df['Tweet_Processed'][ind]))
