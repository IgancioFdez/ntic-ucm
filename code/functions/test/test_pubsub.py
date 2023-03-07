# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:56:46 2023

@author: ifernandez
"""

#https://levelup.gitconnected.com/how-to-use-google-pub-sub-to-build-an-asynchronous-messaging-system-in-python-3b43094627dc
#https://github.com/googleapis/python-pubsub/blob/main/samples/snippets/publisher.py
#http://www.theappliedarchitect.com/setting-up-gcp-pub-sub-integration-with-python/
#pip install google-cloud
#pip install -U google-cloud-pubsub

from google.cloud import pubsub_v1
import os
import logging
import json
import time
from concurrent.futures import TimeoutError

# set key credentials file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'./my-test-project-379108-c38ac66b0cd0.json'

project_id = "my-test-project-379108"
topic_id = "test-project-topic-transformar"
subscription_id="test-project-topic-ntic-normalizar-sub"

publisher = pubsub_v1.PublisherClient()
# The `topic_path` method creates a fully qualified identifier
# in the form `projects/{project_id}/topics/{topic_id}`
topic_path = publisher.topic_path(project_id, topic_id)

subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)


# callback function for processing consumed payloads 
# prints recieved payload
def process_payload(message):
	print(f"Received {message.data}.")
	message.ack()
	print(message.data.decode("utf-8"))

	
# define function that reads from cloud storage
def send_pubsub_message(): 

	#_message1 = "World!"
	#message1 = _message1.encode("utf-8")
	#payload = b'The rain in Wales falls mainly on the snails.'
	
	payload = {"data" : "2022-12-30", "timestamp": time.time()}
	data = json.dumps(payload).encode("utf-8")  
	
	# When you publish a message, the client returns a future.
	future1 = publisher.publish(topic_path, data=data)
	print(future1.result())
	# 3831040081518609
		
# define function that write to cloud storage
def read_pubsub_message(): 
	
	print(f"Listening for messages on {subscription_path}..\n")
	streaming_pull_future = subscriber.subscribe(subscription_path, callback=process_payload)
	# Wrap subscriber in a 'with' block to automatically call close() when done.
	with subscriber:
		try:
			streaming_pull_future.result()
		except TimeoutError:
			streaming_pull_future.cancel()

def app():
	#files=list_cs_files('my-test-project-bucket-ntic')
	while (True):
		send_pubsub_message()
		read_pubsub_message()
	#read_cs_files('my-test-project-bucket-ntic','tweets_usa_es_2022-12-19.json')
	#f=read_cs_files('my-test-project-bucket-ntic','data/json/tweets/raw/2022-12-12/test.json')
	#write_cs_files('my-test-project-bucket-ntic','data/json/tweets/unified/2022-12-12/test.json',f)

if __name__ == '__main__':
	app()