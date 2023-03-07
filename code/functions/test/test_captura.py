# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:56:46 2023

@author: ifernandez
"""
import pandas as pd
import logging
import time
import json
import os
from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1

# set key credentials file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'./my-test-project-379108-c38ac66b0cd0.json'

project_id = "my-test-project-379108"
topic_id = "test-project-topic-ntic"

publisher = pubsub_v1.PublisherClient()
# The `topic_path` method creates a fully qualified identifier
# in the form `projects/{project_id}/topics/{topic_id}`
topic_path = publisher.topic_path(project_id, topic_id)

# define function that reads from cloud storage
def send_pubsub_message(data): 
	
	payload = {"data" : data, "timestamp": time.time()}
	data = json.dumps(payload).encode("utf-8")  
	
	# When you publish a message, the client returns a future.
	future1 = publisher.publish(topic_path, data=data)
	print(future1.result())

	
def app():
	
	aux=-1
	
	df = pd.read_csv('./fechas.csv')
	
	rm=df['activo']!=0
	
	for idx,i in enumerate(rm):
		#print(i)
		if i !=1:
			aux=idx
			break
		
	if aux!=-1:			
		logging.info(aux)
		print(aux)
		print(df.loc[aux,'días'])
		send_pubsub_message(df.loc[aux,'días'])
		df.loc[aux,'activo']=1
		df.to_csv('./fechas.csv', index=False, header=True)

if __name__ == '__main__':
	app()