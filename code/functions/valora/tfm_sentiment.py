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
	text = "Hello, world!"
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