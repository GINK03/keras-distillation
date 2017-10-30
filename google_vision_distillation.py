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
import numpy as np
import os
from PIL import Image
import glob 
import pickle
import sys
import random
import msgpack
import numpy as np
import json

input_tensor = Input(shape=(224, 224, 3))
vgg_model = VGG19(include_top=False, weights='imagenet', input_tensor=input_tensor)
for layer in vgg_model.layers[:9]: # default 15
  layer.trainable = False
x = vgg_model.layers[-1].output 
x = Dropout(0.35)(x)
x = Flatten()(x)
x = Dense(5000, activation='relu')(x)
x = Dropout(0.35)(x)
x = Dense(5000, activation='sigmoid')(x)
model = Model(inputs=vgg_model.input, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam')

def train():
  for i in range(1000):
    print('now iter {} load pickled dataset...'.format(i))
    Xs,ys = [],[]
    names = [name for idx, name in enumerate( glob.glob('./dataset/*.pkl') )]
    random.shuffle( names )
    for idx, name in enumerate(names):
      try:
        X,y = pickle.loads(open(name,'rb').read() ) 
      except EOFError as e:
        continue
      if idx%100 == 0:
        print('now scan iter', idx)
      if idx >= 3000:
        break
      Xs.append( X )
      ys.append( y )

    Xs = np.array( Xs )
    ys = np.array( ys )
    print( Xs.shape )
    print( ys.shape )
    model.fit(Xs, ys, batch_size=16, epochs=1 )
    print('now iter {} '.format(i))
    del Xs
    del ys
    if i%10 == 0:
      model.save_weights('models/{:09d}.h5'.format(i))

def pred():
  tag_index = pickle.loads(open('tag_index.pkl', 'rb').read())
  index_tag = { index:tag for tag, index in tag_index.items() }
  name_img150 = []
  for name in filter(lambda x: '.jpg' in x, sys.argv):
    img = Image.open('{name}'.format(name=name))
    img = img.convert('RGB')
    img150 = np.array(img.resize((150, 150)))
    name_img150.append( (name, img150) )
  model = load_model(sorted(glob.glob('models/*.model'))[-1]) 
  for name, img150 in name_img150:
    result = model.predict(np.array([img150]) )
    result = result.tolist()[0]
    result = { i:w for i,w in enumerate(result)}
    for i,w in sorted(result.items(), key=lambda x:x[1]*-1)[:30]:
      print("{name} tag={tag} prob={prob}".format(name=name, tag=index_tag[i], prob=w) )
if __name__ == '__main__':
  if '--train' in sys.argv:
    train()
  if '--pred' in sys.argv:
    pred()
