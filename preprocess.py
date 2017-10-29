from PIL import Image
import numpy as np
import glob
import sys
import json
import pickle

if '--resize' in sys.argv:
  target_size = (224,224)
  for name in sum( map(lambda x:glob.glob(x), ['images/170911_統合pivot.files/*.png', './images/imgs/*.png', './images/imgs/*.jpg']), []):
    try:
      img = Image.open(name) 
    except OSError as e:
      continue
    img = img.convert('RGB')
    w, h = img.size
    if w > h :
      blank = Image.new('RGB', (w, w))
    if w <= h :
      blank = Image.new('RGB', (h, h))
    blank.paste(img, (0, 0) )
    blank = blank.resize( target_size )

    # Goolge Cloud Vision用データ
    arr = np.asanyarray(blank)
    save_name = hash(arr.tostring())
    
    blank.save('converted_images/{}.jpg'.format(save_name))

if '--make_pkl' in sys.argv:
  
  xy = []
  for name in glob.glob('dataset/*.json'):
    obj = json.loads( open(name).read() )
    description_score = {}
    try:
      for o in obj['responses'][0]['labelAnnotations']:
        description = o['description']
        score = o['score']
        description_score[description] = score
    except KeyError as e:
      continue
    hashnum  = name.split('/').pop().replace('.json', '')
    img = Image.open('converted_images/{}.jpg'.format(hashnum))
    arr = np.asarray(img)
    xy.append( (arr, description_score) )
    print( hashnum, description_score )

  description_index = {}
  for arr, description_score in xy:
    for description, score in description_score.items():
      if description_index.get( description ) is None:
        description_index[ description ] = len(description_index)
 
  Xy = []
  LEN = len(description_index)
  for arr, description_score in xy:
    y = [0.0]*LEN
    for description, score in description_score.items():
      index = description_index[ description ] 
      y[ index ] = score
    Xy.append( (arr, y) )

  open('Xy.pkl', 'wb').write( pickle.dumps(Xy) )
