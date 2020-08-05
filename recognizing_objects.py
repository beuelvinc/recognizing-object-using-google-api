from base64 import b64encode
from oauth2client.client import GoogleCredentials
import googleapiclient.discovery


IMAGE_FILE='road.jpg'
CREDENTIALS_FILE='credential.json'

credentials=GoogleCredentials.from_stream(CREDENTIALS_FILE)
service=googleapiclient.discovery.build("vision",'v1',credentials=credentials)

with open(IMAGE_FILE,'rb') as f:
    image_data=f.read()
    encoded_image_data=b64encode(image_data).decode("utf-8")
	
batch_request=[{
    "image":{"content":encoded_image_data}
,
  "features":[{"type":"LABEL_DETECTION"}]  
}]

request=service.images().annotate(body={"requests":batch_request})
response=request.execute()
if "error" in response:
    print(response['error'])
labels=response['responses'][0]['labelAnnotations']
for label in labels:
    print(label['description'],label['score'])
