# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:56:46 2023

@author: ifernandez
"""

#https://medium.com/google-cloud/automating-google-cloud-storage-management-with-python-92ba64ec8ea8
#https://medium.com/@erdoganyesil/read-file-from-google-cloud-storage-with-python-cf1b913bd134
#https://cloud.google.com/python/docs/reference/storage/latest/buckets
#https://gist.github.com/GabrielSGoncalves/ff9155246c55ead6d33d1103d51bbad1
#https://cloud.google.com/appengine/docs/legacy/standard/python/googlecloudstorageclient/read-write-to-cloud-storage?hl=es-419
#pip install google-cloud-storage

from google.cloud import storage
import os
import json

# set key credentials file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'./my-test-project-379108-c38ac66b0cd0.json'

# define function that reads from cloud storage
def list_cs_files(bucket_name): 
	storage_client = storage.Client()
	#logging.info('Meensaje con ID: ' + storage_client)
	
	file_list = storage_client.list_blobs(bucket_name)
	file_list = [file.name for file in file_list]
	
	#print(file_list)
	return file_list

# define function that reads from cloud storage
def read_cs_files(bucket_name,file_name): 
	storage_client = storage.Client()
	#logging.info('Meensaje con ID: ' + storage_client)
	
	# get bucket with name
	bucket = storage_client.get_bucket(bucket_name)
	# get bucket data as blob
	blob = bucket.get_blob(file_name)
	# convert to string
	#json_data = blob.download_as_string()
	#print(blob.path)
	#print(blob.name)
	#blob.open('r',encoding='utf-8')
	file=blob.download_as_string()
	#print(file) 
	#downloaded_blob = blob.download_as_text()
	#file = file.decode("utf-8")
	file = file.replace(b"'", b'"')
	file = file.decode("utf-8")
	print(file)
	#print(json.loads(file))
	tweets=json.loads(file)
	#return file_list
	for tweet in tweets:
 		print(tweet)
	return tweets
		
# define function that write to cloud storage
def write_cs_files(bucket_name, destination_file_name , source_file_name): 
	storage_client = storage.Client()

	bucket = storage_client.bucket(bucket_name)

	blob = bucket.blob(destination_file_name)
	with open('./data.json', 'w', encoding='utf-8') as outfile:
		outfile.write(json.dumps(source_file_name))
	blob.upload_from_filename(outfile.name)
	os.remove('./data.json')

	return True

def app():
	#files=list_cs_files('my-test-project-bucket-ntic')
	#read_cs_files('my-test-project-bucket-ntic','testdata.xml')
	#read_cs_files('my-test-project-bucket-ntic','tweets_usa_es_2022-12-19.json')
	f=read_cs_files('my-test-project-bucket-ntic','data/json/tweets/raw/2022-12-19/snscrape_tweets_argentina_es_2022-12-19.json')
	#write_cs_files('my-test-project-bucket-ntic','data/tweets/json/unified/2022-12-19/tweets_argentina_es_2022-12-19.json',f)

if __name__ == '__main__':
	app()