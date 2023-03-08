import base64
import functions_framework
import time
import json

# Triggered from a message on a Cloud Pub/Sub topic.
@functions_framework.cloud_event
def app(cloud_event):
	# Print out the data from Pub/Sub, to prove that it worked
	data=base64.b64decode(cloud_event.data["message"]["data"])
	data=data.decode("utf-8")

