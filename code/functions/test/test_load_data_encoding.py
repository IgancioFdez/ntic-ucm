# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:26:14 2022

@author: ifernandez
"""
import pandas as pd 
import json
import re

def app():
	
 	#TWINT
 	lst=[]
 	with open('../../../data/json/tweets/raw/2022-11-18/twint_tweets_corea_es_2022-11-18.json', "r",encoding='utf-8') as file:
		 lines = file.readlines()
 	for line in lines:
		 print(type(line))
		 print(line)
		 y=json.loads(line)
		 print(type(y))
		 print(y)
		 lst.append(y)
		
 	df=pd.DataFrame(lst)
 	print(df.columns)
 	print(df.shape)
 	df=df.loc[0:,['id','date','time','tweet','language','replies_count','retweets_count','likes_count','hashtags','link']]
 	print(df.columns)
 	print(df.shape)
 	print(df.loc[0:,'tweet'])
 	df.to_csv(r'./export_dataframe.csv', index=False,  header=['Tweet_Id', 'Date', 'Time', 'Tweet', 'Language', 'Replies_Count', 'Retweets_Count', 'Likes_Count', 'Hashtags', 'Link', 'Country'],sep='|',encoding='utf-8')

#  	#SNS
# 	lst=[]
# 	with open('./tweets.json', "r",encoding='utf-8') as file:
# 		data = json.dumps(file.read(),ensure_ascii=False)
# 	data=json.loads(data)
# 	data=data.replace("None","\'\'")
# 	data=data.replace("\\n","")
# 	data=data.replace("\"","\'")
# 	data = re.sub("<.*?>","",data)
# 	data = re.sub("card\'.*?cashtags","cashtags",data)
# 	data = re.sub("coordinates.*?date","date",data)
# 	data = re.sub("links.*?media","media",data)
# 	data = re.sub("media.*?mentionedUsers","mentionedUsers",data)
# 	data = re.sub("place.*?quoteCount","quoteCount",data)
# 	data = re.sub("mentionedUsers.*?quoteCount","quoteCount",data)
# 	data = re.sub("inReplyToTweetId.*?inReplyToUser","inReplyToUser",data)
# 	data = re.sub("inReplyToUser.*?lang","lang",data)
# 	data = re.sub("rawContent.*?renderedContent","renderedContent",data)
# 	data=data.replace("[{","{")
# 	data=data.replace("}]","}")
# 	data=data.replace("}, {","}\n{")
# 	data=data.replace("\'", "\"")
# 	data = data.split("\n")
# 	print(len(data))
# 	for i in data:
# 		# parse x:
# 		# getting index of substrings
# 		idx1 = i.index("renderedContent\": \"")
# 		idx2 = i.index("\", \"replyCount\"")

# 		res = i[idx1 + len(" renderedContent :") + 1: idx2]
# 		
# 		res=res.replace("\"", "")
# 		res=res.replace("\\", "")
# 		i=re.sub("renderedContent\": \".*?\", \"replyCount","renderedContent\": \"pre_aux_string\", \"replyCount",i)
# 		i=i.replace("pre_aux_string",res)
# 		
# 		tweet_dict = json.loads(str(i))
# 		aux=tweet_dict["date"].split(" ")
# 		tweet_dict["date"]=aux[0]
# 		tweet_dict["time"]=aux[1].split("+")[0]
# 		lst.append(tweet_dict)

# 	df=pd.DataFrame(lst)
# 	print(df.columns)
# 	print(df.shape)
# 	df=df.loc[0:,['conversationId','date','time','renderedContent','lang','replyCount','retweetCount','likeCount','hashtags','url']]
# 	print(df.columns)
# 	print(df.shape)
# 	print(df.loc[0:,'renderedContent'])
# 	df.to_csv(r'./export_dataframe.csv', index=False,  header=['Tweet_Id', 'Date', 'Time', 'Tweet', 'Language', 'Replies_Count', 'Retweets_Count', 'Likes_Count', 'Hashtags', 'Link'],sep='|',encoding='utf-8')
# 	
if __name__ == '__main__':
	app()