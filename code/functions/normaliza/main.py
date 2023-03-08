# -*- coding: utf-8 -*-

#import os
import base64
import functions_framework
import time
import json
from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1
import pandas as pd
import re
import configparser


# Global properties
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'../test/my-test-project-379108-c38ac66b0cd0.json'
PROPERTIES_FILE_NAME = 'ConfigFile_normalizar.properties'
project_id = "my-test-project-379108"
topic_id = "test-project-topic-transformar"
subscription_id="test-project-topic-ntic-normalizar-sub"

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_id)

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
	column_names=["Tweet_Id","Date","Time","Tweet","Language","Mentions","Replies_Count","Retweets_Count","Likes_Count","Hashtags","Cashtags","Link","Country"]
	df=pd.DataFrame(data=None,columns=column_names,index=None)
	
	try:
		print("[TWINT] Ruta Fichero : "+path_file_twint)
		with open(path_file_twint, "r",encoding='utf-8') as file:
			lines = file.readlines()
			#print("["+path_file_twint+"]"+"Tweets en el fichero : "+str(len(lines)))
			cont=0
			for line in lines:
				#data = file.read()
				yrs_string=line.rstrip()
				#print(yrs_string)
				tweet_dict=json.loads(yrs_string)
				#print(tweet_dict)
				if (language==tweet_dict["language"]):
					#print(tweet_dict["conversation_id"])
					#print(tweet_dict["created_at"])
					#print(tweet_dict["date"])
					#print(tweet_dict["time"])
					#print(tweet_dict["tweet"])
					#print(tweet_dict["language"])
					#print(tweet_dict["mentions"])
					#print(tweet_dict["replies_count"])
					#print(tweet_dict["retweets_count"])
					#print(tweet_dict["likes_count"])
					#print(tweet_dict["hashtags"])
					#print(tweet_dict["cashtags"])
					#print(tweet_dict["link"])
					
					list = [tweet_dict["conversation_id"],tweet_dict["date"],tweet_dict["time"],str(tweet_dict["tweet"]),tweet_dict["language"],str(tweet_dict["mentions"]),tweet_dict["replies_count"],tweet_dict["retweets_count"],tweet_dict["likes_count"],str(tweet_dict["hashtags"]),tweet_dict["cashtags"],str(tweet_dict["link"]),str(country)]
					df.loc[len(df)] = list
					cont +=1
					
			#print(df)
			if cont>0:
				with open(path_file_result, 'w', encoding='utf-8') as file:
					df.to_json(file, orient='records', force_ascii=False)
			#print("["+path_file_twint+"]"+"Tweets válidos : "+str(cont))
			
	except Exception:
		print("[SNS] Ruta Fichero : "+path_file_sns)
		try:
			with open(path_file_sns, "r",encoding='utf-8') as file:
				data = json.dumps(file.read())
				#print(data)
				#print(type(data))
				data=json.loads(data)
				#print(data)
				data=data.replace("None","''")
				data = re.sub("<.*?>","",data)
				data = re.sub("card.*?cashtags","cashtags",data)
				data = re.sub("links.*?media","media",data)
				data = re.sub("media.*?mentionedUsers","mentionedUsers",data)
				#data = re.sub("TextLink(.*?)","",data)
				#data = re.sub("SummaryCard(.*?)","",data)
				data=data.replace("[{","{")
				data=data.replace("}]","}")
				data=data.replace("}, {","}|{")
				data=data.replace('\\n','')
				data=data.replace('\"','')
				data=data.replace('\'','\"')
				#print(data)
				#print(yrs_string)
				data = data.split("|")
				
				#print("["+path_file_sns+"]"+"Tweets en el fichero : "+str(len(data)))
				cont=0
				for i in data:
					# parse x:
					tweet_dict = json.loads(str(i))

					if (language==tweet_dict["lang"]):
						#print(tweet_dict["conversationId"])
						aux=tweet_dict["date"].split(" ")
						date=aux[0]
						time=aux[1].split("+")[0]
						#print(tweet_dict["date"])
						#print(date)
						#print(time)
						#print(tweet_dict["rawContent"])
						#print(tweet_dict["lang"])
						#print(tweet_dict["mentionedUsers"])
						#print(tweet_dict["replyCount"])
						#print(tweet_dict["retweetCount"])
						#print(tweet_dict["likeCount"])
						#print(tweet_dict["hashtags"])
						#print(tweet_dict["cashtags"])
						#print(tweet_dict["url"])
						
						list = [str(tweet_dict["conversationId"]),date,time,str(tweet_dict["renderedContent"]).encode('utf8'),tweet_dict["lang"],str(tweet_dict["mentionedUsers"]),tweet_dict["replyCount"],tweet_dict["retweetCount"],tweet_dict["likeCount"],str(tweet_dict["hashtags"]),tweet_dict["cashtags"],str(tweet_dict["url"]),str(country)]
						df.loc[len(df)] = list
						cont +=1
						
				#print(df)
				if cont>0:
					with open(path_file_result, 'w', encoding='utf-8') as file:
						df.to_json(file, orient='records', force_ascii=False)
				#print("["+path_file_sns+"]"+"Tweets válidos : "+str(cont))

		except Exception:
			print('File does not exist')

# Function to load tweets from file		
def normalizeTweetsFiles(props,origin_path,destination_path,folder):
	for country in props["teams"]:
		for language in props["teams"][country]["languages"]:
			path_file_twint=origin_path+"/"+"twint_tweets_"+country+"_"+language+"_"+folder+".json"
			path_file_sns=origin_path+"/"+"snscrape_tweets_"+country+"_"+language+"_"+folder+".json"
			path_file_result=destination_path+"/"+"tweets_"+country+"_"+language+"_"+folder+".json"
			#print(path_file_twint)
			#print(path_file_sns)
			#print(path_file_result)

			#get_data_from_file(path_file_twint,path_file_sns,path_file_result,language,country)
		
# Function that sends to topic
def send_pubsub_message(data): 
	data=json.loads(data)
	data=data["data"]
	payload = {"data" : data, "timestamp": time.time()}
	data = json.dumps(payload).encode("utf-8")  
	# When you publish a message, the client returns a future.
	future1 = publisher.publish(topic_path, data=data)
	#print(future1.result())

@functions_framework.cloud_event
def app(cloud_event):			              
#def app():
	# Print out the data from Pub/Sub, to prove that it worked
	data=base64.b64decode(cloud_event.data["message"]["data"])
	data=data.decode("utf-8")

	folder=data

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
	#normalizeTweetsFiles(props,origin_path,destination_path,folder)

	#Send message to topic
	send_pubsub_message(folder)

# if __name__ == '__main__':
# 	app()