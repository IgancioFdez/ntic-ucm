# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:56:46 2023

@author: ifernandez
"""

#https://cloud.google.com/bigquery/docs/samples/bigquery-table-insert-rows?hl=es-419#bigquery_table_insert_rows-python
#https://blog.morizyun.com/python/library-bigquery-google-cloud.html
#https://blog.coupler.io/how-to-crud-bigquery-with-python/
#pip install google-cloud-bigquery

from google.cloud import bigquery
import os

# set key credentials file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'./my-test-project-379108-c38ac66b0cd0.json'

def export_items_to_bigquery():
    # Instantiates a client
    bigquery_client = bigquery.Client()

    # Prepares a reference to the dataset
    dataset_ref = bigquery_client.dataset('tweets_wc_ntic_master')

    table_ref = dataset_ref.table('tweets_raw')
    table = bigquery_client.get_table(table_ref)  # API call

    rows_to_insert = [
        ("1594116409549651968",
		 "2022-11-20",
		 "Luis Enrique si españa gana el mundial",
		 "es",
		 2,
		 0,
		 12,
		 "['mundialqatar2022', 'mundialenqatar', 'qatar2022', 'qatarworldcup2022']",
		 "https:\/\/twitter.com\/tivotabo\/status\/1594116409549651968",
		 "españa",
		 0.0)
    ]
    errors = bigquery_client.insert_rows(table, rows_to_insert)  # API request
    assert errors == []

def exist_record():
	bigquery_client = bigquery.Client()

	query = ('SELECT tweet_id FROM `{}.{}.{}` WHERE tweet_id="{}" LIMIT 1'
            .format('my-test-project-379108', 'tweets_wc_ntic_master', 'tweets_raw', '1594116409549651968'))

	try:
		query_job = bigquery_client.query(query)
		query_job.result()
	except Exception as e:
		print("Error")
		print(e)

	return False

def update_record():
    bigquery_client = bigquery.Client()

    query = ('UPDATE `{}.{}.{}` SET tweet_sentiment={} WHERE tweet_id="{}"'
            .format('my-test-project-379108', 'tweets_wc_ntic_master', 'tweets_raw',1.0, '1594116409549651968'))

    try:
        query_job = bigquery_client.query(query)
        is_exist = len(list(query_job.result())) >= 1
        print('Exist id: {}'.format('1594116409549651968') if is_exist else 'Not exist id: {}'.format('1594116409549651968'))
        return is_exist
    except Exception as e:
        print("Error")
        print(e)

    return False

def app():
	export_items_to_bigquery()
	exist_record()
	update_record()

if __name__ == '__main__':
	app()