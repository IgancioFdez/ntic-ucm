# -*- coding: utf-8 -*-


import configparser
import nest_asyncio
nest_asyncio.apply()
__import__('IPython').embed()
import twint
import datetime
import snscrape.modules.twitter as sntwitter
from datetime import datetime, timedelta
from google.cloud import storage
import functions_framework
import pandas as pd
import logging
import time
import json
import os
from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1
 

PROPERTIES_FILE_NAME = 'ConfigFile.properties'
project_id = "my-test-project-379108"
topic_id = "test-project-topic-ntic-normalizar"
bucket_id = "my-test-project-bucket-ntic"
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_id)

# Function to load properties
def get_properties(name):
 	#INIT#
 	c = configparser.RawConfigParser()
 	c.read(name, encoding='utf-8')
 	props={}
 	props["global_hastags"]=str(c.get('GlobalTagsSection', 'gltags.edition')+c.get('GlobalTagsSection', 'gltags.tournament')+c.get('GlobalTagsSection', 'gltags.organizer')).replace(";", " OR ")
 	props["countries"]=str(c.get('GlobalTagsSection', 'gltags.countries')).split(";")
 	props["limit"]=c.get('GlobalTagsSection', 'gltags.limit')
 	props["likes"]=c.get('GlobalTagsSection', 'gltags.numlikes')
 	props["data_path"]=c.get('GlobalTagsSection', 'gltags.data.path')
 	props["destination_folder"]=c.get('GlobalTagsSection', 'gltags.destination.folder')
 	teams={}
 	for country in props["countries"]:
 		 team={}
 		 team["name"]=str(c.get('TeamsTagsSection', 'tmtags.'+country+'.team'))
 		 team["tags"]=str(c.get('TeamsTagsSection', 'tmtags.'+country+'.federation')+c.get('TeamsTagsSection', 'tmtags.'+country+'.names')+c.get('TeamsTagsSection', 'tmtags.'+country+'.tags')).replace(";", " OR ")
 		 team["languages"]=str(c.get('TeamsTagsSection', 'tmtags.'+country+'.languages')).split(";")
 		 team["active"]=str(c.get('TeamsTagsSection', 'tmtags.'+country+'.active'))
 		 team["group"]=str(c.get('TeamsTagsSection', 'tmtags.'+country+'.group'))
 		 teams[country]=team 
 	props["teams"]=teams
 	return props

# Function to get tweets using Twint scrapper
def twint_get_tweets(query,language,file,limit,numlikes,since,until):
	c = twint.Config()
	c.Search = str(query)
	c.Lang = str(language)
	c.Limit = limit
	c.Hashtags = True
	c.Store_json = True
	c.Filter_retweets = True
	c.Output = file
	c.Min_likes = numlikes
	c.Since = str(since) 
	c.Until = str(until)
	twint.run.Search(c)
	return c.Output

# Function to get tweets using SNS scrapper
def sns_get_tweets(query,language,file,limit,numlikes,since,until):
 	query = query + " min_faves:"+str(numlikes)+ " since:"+str(since)+" until:"+str(until)+" lang:"+language+" -filter:replies"
 	#print(query)
 	tweets = []
 	for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
		 if i < limit:
			 #print(vars(tweet))
			 #tweets.append(json.dumps(tweet.__dict__))
			 tweets.append(vars(tweet))
		 else:
			 break
 	# convert into JSON:
	y = json.dumps(tweets,indent=4, sort_keys=True, default=str, ensure_ascii=False)
	return y

# Function that write to cloud storage
def write_cs_files(bucket_name, destination_file_name , file): 
 	storage_client = storage.Client()
 	bucket = storage_client.bucket(bucket_name)
 	blob = bucket.blob(destination_file_name)
 	with open('./json_data.json', 'w', encoding='utf-8') as outfile:
		outfile.write(json.dumps(file))
 	blob.upload_from_filename(outfile.name)
 	os.remove('./json_data.json')
 	return True

# Function to get tweets
def get_tweets(props,scrapper_path):
 	d = datetime.now() - timedelta(days=1)
 	for country in props["teams"]:
		if (props["teams"][country]["active"])=="true":
 			for language in props["teams"][country]["languages"]:
 			query="("+props["global_hastags"]+")"+" "+"("+props["teams"][country]["tags"]+")"+"  -bitcoin -nft -token"
 			#print(query)
 			since = d.strftime('%Y-%m-%d')
 			until = datetime.now().strftime('%Y-%m-%d')
 			file_twint=scrapper_path+"/"+"twint_tweets_"+country+"_"+language+"_"+since+".json"
 			file_sns=scrapper_path+"/"+"snscrape_tweets_"+country+"_"+language+"_"+since+".json"
 			f=twint_get_tweets(query,language,file_twint,100000,5,since,until)
			write_cs_files(bucket_id,'data/json/tweets/raw/'+since+'/'+'twint_tweets_'+country+'_'+language+'_'+since+'.json',f)
 			f=sns_get_tweets(query,language,file_sns,100000,5,since,until)
			write_cs_files(bucket_id,'data/json/tweets/raw/'+since+'/'+'snscrape_tweets_'+country+'_'+language+'_'+since+'.json',f)

# Function that sends to topic
def send_pubsub_message(data): 
 	payload = {"data" : data, "timestamp": time.time()}
 	data = json.dumps(payload).encode("utf-8")  

 	# When you publish a message, the client returns a future.
 	future1 = publisher.publish(topic_path, data=data)
 	#print(future1.result())
	
@functions_framework.http
def app(request):

 	#Get properties
 	props=get_properties(PROPERTIES_FILE_NAME)
 	#print(props)

 	d = datetime.now() - timedelta(days=1)
 	since=d.strftime('%Y-%m-%d')
 	# print(since)
 	# print(datetime.now().strftime('%Y-%m-%d'))
 	
 	#Create directory structure for tweets json files
 	#parent_path = os.getcwd()
	destination_path =props["data_path"]+props["destination_folder"]+since
	scrapper_path =destination_path.replace("\\","/")
#  	#print ("The current working directory is %s" % parent_path)
#  	# print ("The json destination directory is %s" % destination_path)
#  	# print ("The scrapper directory is %s" % scrapper_path)
#  	try:
# 		os.mkdir(destination_path)
#  	except OSError:
# 		print ("Creation of the directory %s failed" % str(destination_path))
#  	else:
# 		print ("Successfully created the directory %s " % str(destination_path))
 			
 	#Get tweets
 	get_tweets(props,scrapper_path)

	#Send message to topic
	send_pubsub_message(since)
		
	return '{"status":"200", "data": "OK"}'
