# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 09:33:25 2023

@author: ifernandez
"""
#https://medium.com/google-cloud/sentiment-analysis-using-google-cloud-machine-learning-552be9b9c39b
#https://medium.com/google-cloud/sentiment-analysis-using-natural-language-processing-nlp-api-in-google-cloud-b87b5ec4d388
#https://cloud.google.com/natural-language/docs/sentiment-analysis-client-libraries?hl=es-419
#pip install --upgrade google-cloud-language

# Imports the Google Cloud client library
from google.cloud import language_v1
import os

# set key credentials file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'./my-test-project-379108-c38ac66b0cd0.json'

# Instantiates a client
client = language_v1.LanguageServiceClient()

def app():
    
	# The text to analyze
	text = "Vamos españa"
	document = language_v1.Document(
	    content=text, type_=language_v1.Document.Type.PLAIN_TEXT
	)

	# Detects the sentiment of the text
	sentiment = client.analyze_sentiment(
	    request={"document": document}
	).document_sentiment

	print("Text: {}".format(text))
	print("Sentiment: {}, {}".format(sentiment.score, sentiment.magnitude))

if __name__ == '__main__':
	app()