# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:26:14 2022

@author: ifernandez
"""

import utils.load_properties as configuration
#import utils.mongo_functions as bd
import utils.twitter_scrapper_twint as twint
import utils.twitter_scrapper_snscrape as sns
import os
from datetime import datetime, timedelta
import sys


PROPERTIES_FILE_NAME = 'ConfigFile.properties'

def get_tweets(props,scrapper_path,cli_basic,cli_country):
	
	d = datetime.now() - timedelta(days=1)
	
	if (cli_country!="all"):
		print(cli_country)
		if (props["teams"][cli_country]["active"])=="true":
	           #print(props["teams"][country])
	           
			for language in props["teams"][cli_country]["languages"]:
	               #print(props["teams"][country]['name'])
				query="("+props["global_hastags"]+")"+" "+"("+props["teams"][cli_country]["tags"]+")"+"  -bitcoin -nft -token"
				file_twint=scrapper_path+"/"+"twint_tweets_"+cli_country+"_"+language+"_"+(d).strftime('%Y-%m-%d')+".json"
				file_sns=scrapper_path+"/"+"snscrape_tweets_"+cli_country+"_"+language+"_"+(d).strftime('%Y-%m-%d')+".json"
				print(query)
				since = datetime(2022, 12, 19).strftime('%Y-%m-%d')
				until = datetime(2022, 12, 20).strftime('%Y-%m-%d')
				if (cli_basic=="true"):
					sns.get_tweets_aux(query,language,file_sns,100000,5,since,until)
				else:
					twint.get_tweets_aux(query,language,file_twint,100000,5,since,until)
	else:
		for country in props["teams"]:
	        #print(country)
	        
			if (props["teams"][country]["active"])=="true":
	            #print(props["teams"][country])
	            
				for language in props["teams"][country]["languages"]:
	                #print(props["teams"][country]['name'])
					query="("+props["global_hastags"]+")"+" "+"("+props["teams"][country]["tags"]+")"+"  -bitcoin -nft -token"
					file_twint=scrapper_path+"/"+"twint_tweets_"+country+"_"+language+"_"+(d).strftime('%Y-%m-%d')+".json"
					file_sns=scrapper_path+"/"+"snscrape_tweets_"+country+"_"+language+"_"+(d).strftime('%Y-%m-%d')+".json"
					print(query)
					since = datetime(2022, 12, 19).strftime('%Y-%m-%d')
					until = datetime(2022, 12, 20).strftime('%Y-%m-%d')
					if (cli_basic=="true"):
						sns.get_tweets_aux(query,language,file_sns,100000,5,since,until)
					else:
						twint.get_tweets_aux(query,language,file_twint,100000,5,since,until)
                
                
def app():
	cli_basic=sys.argv[1]
	cli_country=sys.argv[2]
    
    #Get properties for twitter scraper
	c=configuration.init(PROPERTIES_FILE_NAME)
	props=configuration.getCompetitionProperties(c)
	props["teams"]=configuration.getTeamsProperties(c,props)
	print(props)


    #Create directory structure for tweets json files
	parent_path = os.getcwd()
	destination_path=props["destination_folder"]+(datetime.now()).strftime('%Y-%m-%d')
	scrapper_path = '.'+destination_path.replace("\\","/")
	print ("The current working directory is %s" % parent_path)
	print ("The json destination directory is %s" % parent_path+destination_path)
	print ("The scrapper directory is %s" % scrapper_path)
	try:
		os.mkdir(parent_path+destination_path)
	except OSError:
		print ("Creation of the directory %s failed" % str(parent_path+destination_path))
	else:
		print ("Successfully created the directory %s " % str(parent_path+destination_path))
        
        
    #Get tweets
	get_tweets(props,scrapper_path,cli_basic,cli_country)


if __name__ == '__main__':
    app()