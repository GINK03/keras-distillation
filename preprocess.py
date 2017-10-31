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
# 余計なフィイルを消す
import os
if '--remove_extra_pickle' in sys.argv:
  for name in glob.glob('./dataset/*.pkl'):
    os.remove(name)

# jsonで頻出タグを検索して穴埋め
from collections import Counter
if '--search_top_5000' in sys.argv:
  descs = []
  for index, name in enumerate(glob.glob('./vision/pexels*.json')):
    print(index, name)
    if index > 100000:
      break
    obj = json.loads( open(name).read() )
    description_score = {}
    try:
      for o in obj['responses'][0]['labelAnnotations']:
        description = o['description']
        score = o['score']
        description_score[description] = score
    except KeyError as e:
      continue
    #print(description_score) 
    for desc in description_score.keys():
      descs.append(desc)
  descs_freq = dict(Counter(descs))
  desc_index = {}
  for index, (desc, freq) in enumerate(sorted( descs_freq.items(), key=lambda x:x[1]*-1)[:5000]):
    print(desc, freq)
    desc_index[desc] = index
  open('desc_index.json','w').write( json.dumps(desc_index, indent=2, ensure_ascii=False) )

if '--make_pkl' in sys.argv:
  target_size = (224,224)
  desc_index = json.loads( open('desc_index.json').read())
  for img_name in glob.glob('download/pexels*.jpeg'):
    if not os.path.exists(img_name):
      continue

    save_name = 'dataset/{}.pkl'.format(img_name.split('/').pop() )
    json_name = 'vision/{}.json'.format(img_name.split('/').pop() )
    obj = json.loads( open(json_name).read() )
    description_score = {}
    try:
      for o in obj['responses'][0]['labelAnnotations']:
        description = o['description']
        score = o['score']
        description_score[description] = score
    except KeyError as e:
      print(e)
      continue
    if os.path.exists(save_name):
      continue
    img = Image.open(img_name)
    try:
      img = img.convert('RGB')
    except OSError as e:
      continue
  
    w, h = img.size
    if w > h :
      blank = Image.new('RGB', (w, w))
    if w <= h :
      blank = Image.new('RGB', (h, h))
    blank.paste(img, (0, 0) )
    blank = blank.resize( target_size )
    X = np.asanyarray(blank)
    y = [0.0]*len(desc_index)
    for desc, score in description_score.items():
      index = desc_index.get(desc)
      if index is None:
        continue
      y[index] = float(score)
    y = np.array(y)
    print(X.shape)
    print(y.shape)
    print(y)

