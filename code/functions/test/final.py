# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 12:56:22 2023

@author: ifernandez
"""
from google.cloud import bigquery
from google.cloud import language_v1
import os

# set key credentials file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'./my-test-project-379108-c38ac66b0cd0.json'

# Instantiates a client
client = language_v1.LanguageServiceClient()

def get_sentiment(text):
	document = language_v1.Document(
	    content=text, type_=language_v1.Document.Type.PLAIN_TEXT
	)

	# Detects the sentiment of the text
	sentiment = client.analyze_sentiment(
	    request={"document": document}
	).document_sentiment

	print("Text: {}".format(text))
	print("Sentiment: {}, {}".format(sentiment.score, sentiment.magnitude))
	return round(sentiment.score, 4)

def export_items_to_bigquery(s):
    # Instantiates a client
    bigquery_client = bigquery.Client()

    # Prepares a reference to the dataset
    dataset_ref = bigquery_client.dataset('tweets_wc_ntic_master')

    table_ref = dataset_ref.table('tweets_raw')
    table = bigquery_client.get_table(table_ref)  # API call

    rows_to_insert = [
        ("1594116409549651968",
		 "2022-11-20",
		 "Luis Enrique si españa gana el mundial",
		 "es",
		 2,
		 0,
		 12,
		 "['mundialqatar2022', 'mundialenqatar', 'qatar2022', 'qatarworldcup2022']",
		 "https:\/\/twitter.com\/tivotabo\/status\/1594116409549651968",
		 "españa",
		 s)
    ]
    errors = bigquery_client.insert_rows(table, rows_to_insert)  # API request
    assert errors == []

def app():
	# The text to analyze
	text = "Vamos españa es la mejor"
	s=get_sentiment(text)
	print(s)
	export_items_to_bigquery(s)

if __name__ == '__main__':
	app()