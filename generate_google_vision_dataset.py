import requests
import json
import base64
import os
import glob
import concurrent.futures 

GOOGLE_CLOUD_VISION_API_URL = 'https://vision.googleapis.com/v1/images:annotate?key='
API_KEY = os.environ['GOOGLE']
def goog_cloud_vison(image_content):
    api_url = GOOGLE_CLOUD_VISION_API_URL + API_KEY
    req_body = json.dumps({
        'requests': [{
            'image': {
                'content': image_content.decode()
            },
            'features': [{
                'type': 'LABEL_DETECTION',
                'maxResults': 100,
            }]
        }]
    })
    res = requests.post(api_url, data=req_body)
    return res.json()

def img_to_base64(filepath):
    with open(filepath, 'rb') as img:
        img_byte = img.read()
    return base64.b64encode(img_byte)

def mapper( name ):
  try:
    save_name = 'dataset/' + name.split('/').pop().replace('.jpg', '').replace('.png', '') + '.json'
    if os.path.exists(save_name) is True:
      return None
    img = img_to_base64(name)
    res_json = goog_cloud_vison(img)
    raw_obj = json.dumps( res_json, indent=2 ) 
    print( raw_obj )
    open( save_name, 'w').write( raw_obj )
  except Exception as e:
    print(e) 

names = [name for name in glob.glob('converted_images/*')]

with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
  executor.map( mapper, names )
