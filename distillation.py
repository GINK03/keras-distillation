import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Activation, Dropout, Flatten, Dense, Reshape, merge
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.core import Dropout
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.optimizers import SGD, Adam
import numpy as np
import os
from PIL import Image
import glob 
import pickle
import sys
import random
import re
import numpy as np
import json
from gzip import decompress
input_tensor = Input(shape=(224, 224, 3))
vgg_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
for layer in vgg_model.layers[:6]: # default 15
  layer.trainable = False
x = vgg_model.layers[-1].output 
x = Flatten()(x)
x = BN()(x)
x = Dense(5000, activation='relu')(x)
x = Dropout(0.35)(x)
x = Dense(5000, activation='sigmoid')(x)
model = Model(inputs=vgg_model.input, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam')

def train():
  num = -1
  gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
  if gpu is None:
    gpu = '0'
  print('Using GPU numnber is', gpu)
  try:
    ''' 古いデータからリカバー '''
    latest_file = sorted(glob.glob('models{}/*.h5'.format(gpu)) ).pop()
    num = int( re.search(r'(\d{1,}).h5', latest_file).group(1) )
    print('load')
    model.load_weights(latest_file)
    print('loaded model', latest_file, num)
  except Exception as e:
    if str(e) != 'pop from empty list':
      print('Error', e)
      sys.exit()

  for i in range(1000):
    if i <= num:
      continue
    print('now iter {} load pickled dataset...'.format(i))
    Xs,ys = [],[]
    names = glob.glob('../image-free-download-scraper/dataset/*.pkl')
    #random.shuffle( names )
    for idx, name in enumerate( random.sample(names, 5000) ):
      if idx%100 == 0:
        print('now scan iter', idx)
      
      try:
        X,y = pickle.loads( decompress(open(name,'rb').read()) ) 
      except EOFError as e:
        continue
      Xs.append( X )
      #print(y)
      ys.append( y )

    Xs = np.array( Xs, dtype=np.float32 )
    ys = np.array( ys, dtype=np.float32 )
    print( Xs.shape )
    print( ys.shape )
    print(ys)
    model.fit(Xs, ys, batch_size=16, epochs=1 )
    print('now iter {} '.format(i))
    
    model.save_weights('models{gpu}/{i:09d}.h5'.format(gpu=gpu, i=i) )
    try:
      os.remove('models{gpu}/{i:09d}.h5'.format(gpu=gpu, i=i-2))
    except FileNotFoundError as e:
      ...

def pred():
  tag_index = json.loads(open('./tag_index.json', 'r').read())
  index_tag = { index:tag for tag, index in tag_index.items() }
  Xs,ys = [],[]
  for name in filter(lambda x: '.pkl' in x, sys.argv):
    print(name)
    X,y = pickle.loads( decompress(open(name,'rb').read()) ) 
    Xs.append(X)
    ys.append(y)
  Xs,ys = np.array(Xs, dtype=np.float32),np.array(ys,dtype=np.float32)
  print(sys.argv)
  use_model = sorted(glob.glob('models0/*.h5'))[-1]
  print('use model', use_model)
  model.load_weights(use_model) 
  
  result = model.predict( Xs )
  result = result.tolist()[0]
  print(result)
  print(ys.tolist()[0])
  result = { i:w for i,w in enumerate(result)}
  for i,w in sorted(result.items(), key=lambda x:x[1]*-1)[:30]:
    print("{name} tag={tag} prob={prob}, r".format(name=name, tag=index_tag[i].replace(' ', '-'), prob=w) )
  sys.exit()

if __name__ == '__main__':
  if '--train' in sys.argv:
    train()
  if '--pred' in sys.argv:
    pred()
