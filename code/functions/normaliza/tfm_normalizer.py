# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:26:14 2022

@author: ifernandez
"""
import os
import base64
#import functions_framework
import time
import json
#from concurrent.futures import TimeoutError
#from google.cloud import pubsub_v1
import pandas as pd
import re
import configparser
#from google.cloud import storage


# Global properties
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'../test/my-test-project-379108-c38ac66b0cd0.json'
PROPERTIES_FILE_NAME = 'ConfigFile_normalizar.properties'
project_id = "my-test-project-379108"
topic_id = "test-project-topic-transformar"
subscription_id="test-project-topic-ntic-normalizar-sub"

# publisher = pubsub_v1.PublisherClient()
# topic_path = publisher.topic_path(project_id, topic_id)

# Function to load properties
def get_properties(name):
	config = configparser.RawConfigParser()
	config.read(name, encoding='utf-8')
	props={}
	props["countries"]=str(config.get('GlobalTagsSection', 'gltags.countries')).split(";")
	props["data_path"]=config.get('GlobalTagsSection', 'gltags.data.path')
	props["origin_folder"]=config.get('GlobalTagsSection', 'gltags.origin.folder')
	props["destination_folder"]=config.get('GlobalTagsSection', 'gltags.destination.folder')
	teams={}
	for country in props["countries"]:
		team={}
		team["name"]=str(config.get('TeamsTagsSection', 'tmtags.'+country+'.team'))
		team["languages"]=str(config.get('TeamsTagsSection', 'tmtags.'+country+'.languages')).split(";")
		teams[country]=team 
	props["teams"]=teams
	return props

# Function to load tweets from file
def get_data_from_file(path_file_twint,path_file_sns,path_file_result,language,country):
	try:
		print("[TWINT] Ruta Fichero : "+path_file_twint)
		lst=[]
		with open(path_file_twint, "r",encoding='utf-8') as file:
			lines = file.readlines()
		for line in lines:
	# 		print(type(line))
	# 		print(line)
			tweet_dict=json.loads(line)
	# 		print(type(y))
	# 		print(y)
			if (language==tweet_dict["language"]):
				tweet_dict["country"]=country
				lst.append(tweet_dict)
				
		if len(lst)>0:
			df=pd.DataFrame(lst)
			df=df.loc[0:,['conversation_id','date','time','tweet','language','replies_count','retweets_count','likes_count','hashtags','link','country']]
# 			print(df.columns)
# 			print(df.shape)
# 			print(df.loc[0:,'tweet'])
			df.to_csv(path_file_result, index=False,  header=['Tweet_Id', 'Date', 'Time', 'Tweet', 'Language', 'Replies_Count', 'Retweets_Count', 'Likes_Count', 'Hashtags', 'Link', 'Country'],sep='|',encoding='utf-8')
	
	except Exception:
		print("[SNS] Ruta Fichero : "+path_file_sns)
		try:
			lst=[]
			with open(path_file_sns, "r",encoding='utf-8') as file:
				data = json.dumps(file.read(),ensure_ascii=False)
			data=json.loads(data)
			data=data.replace("None","\'\'")
			data=data.replace("\\n","")
			data=data.replace("\"","\'")
			data = re.sub("<.*?>","",data)
			data = re.sub("card\'.*?cashtags","cashtags",data)
			data = re.sub("coordinates.*?date","date",data)
			data = re.sub("links.*?media","media",data)
			data = re.sub("media.*?mentionedUsers","mentionedUsers",data)
			data = re.sub("place.*?quoteCount","quoteCount",data)
			data = re.sub("mentionedUsers.*?quoteCount","quoteCount",data)
			data = re.sub("inReplyToTweetId.*?inReplyToUser","inReplyToUser",data)
			data = re.sub("inReplyToUser.*?lang","lang",data)
			data = re.sub("rawContent.*?renderedContent","renderedContent",data)
			data=data.replace("[{","{")
			data=data.replace("}]","}")
			data=data.replace("}, {","}\n{")
			data=data.replace("\'", "\"")
			data = data.split("\n")
			#print(len(data))
			for i in data:
				# parse x:
				# getting index of substrings
				idx1 = i.index("renderedContent\": \"")
				idx2 = i.index("\", \"replyCount\"")
				res = i[idx1 + len(" renderedContent :") + 1: idx2]
				res=res.replace("\"", "")
				res=res.replace("\\", "")
				i=re.sub("renderedContent\": \".*?\", \"replyCount","renderedContent\": \"pre_aux_string\", \"replyCount",i)
				i=i.replace("pre_aux_string",res)
				tweet_dict = json.loads(str(i))
				if (language==tweet_dict["lang"]):
					aux=tweet_dict["date"].split(" ")
					tweet_dict["date"]=aux[0]
					tweet_dict["time"]=aux[1].split("+")[0]
					tweet_dict["country"]=country
					lst.append(tweet_dict)
			
			if len(lst)>0:
				df=pd.DataFrame(lst)
				df=df.loc[0:,['conversationId','date','time','renderedContent','lang','replyCount','retweetCount','likeCount','hashtags','url','country']]
# 				print(df.columns)
# 				print(df.shape)
# 				print(df.loc[0:,'renderedContent'])
				df.to_csv(path_file_result, index=False,  header=['Tweet_Id', 'Date', 'Time', 'Tweet', 'Language', 'Replies_Count', 'Retweets_Count', 'Likes_Count', 'Hashtags', 'Link', 'Country'],sep='|',encoding='utf-8')
			
		except Exception:
			print('File does not exist')

# Function to load tweets from file		
def normalizeTweetsFiles(props,origin_path,destination_path,folder):
	for country in props["teams"]:
		for language in props["teams"][country]["languages"]:
			path_file_twint="\'../../.."+origin_path+"/"+"twint_tweets_"+country+"_"+language+"_"+folder+".json"+"'"
			path_file_sns="\'../../.."+origin_path+"/"+"snscrape_tweets_"+country+"_"+language+"_"+folder+".json"+"'"
			path_file_result="r'../../.."+destination_path+"/"+"tweets_"+country+"_"+language+"_"+folder+".csv"+"'"
# 			print(path_file_twint)
# 			print(path_file_sns)
# 			print(path_file_result)

			get_data_from_file(path_file_twint,path_file_sns,path_file_result,language,country)
		
# Function that sends to topic
# def send_pubsub_message(data): 
# 	data=json.loads(data)
# 	data=data["data"]
# 	payload = {"data" : data, "timestamp": time.time()}
# 	data = json.dumps(payload).encode("utf-8")  
# 	# When you publish a message, the client returns a future.
# 	future1 = publisher.publish(topic_path, data=data)
# 	#print(future1.result())

#@functions_framework.cloud_event
#def app(cloud_event):			              
def app():
	# Print out the data from Pub/Sub, to prove that it worked
# 	data=base64.b64decode(cloud_event.data["message"]["data"])
# 	data=data.decode("utf-8")

	folder='2022-11-18'

	#Get properties
	props=get_properties(PROPERTIES_FILE_NAME)
	#print(props)

	#Create directory structure for tweets files normalized
	#parent_path = os.getcwd()
	#print(parent_path)
	#origin_path=props["data_path"]+props["origin_folder"]+props["folder_name"]
	origin_path=props["origin_folder"]+folder
	origin_path =origin_path.replace("\\","/")
	#print(origin_path)
	#destination_path ='..\..\..'+props["destination_folder"]+folder
	#print(destination_path)

	# 	try:
	# 		os.mkdir(destination_path)
	# 	except OSError:
	# 		print ("Creation of the directory %s failed" % str(destination_path))
	# 	else:
	# 		print ("Successfully created the directory %s " % str(destination_path))

	#destination_path = props["data_path"]+props["destination_folder"]+folder
	destination_path = props["destination_folder"]+folder
	destination_path =destination_path.replace("\\","/")
	#print(destination_path)
	#print ("The current working directory is %s" % parent_path)
	#print ("The normalized json destination directory is %s" % destination_path)

	#Normalize tweets files
	normalizeTweetsFiles(props,origin_path,destination_path,folder)

	#Send message to topic
	#send_pubsub_message(folder)

if __name__ == '__main__':
 	app()