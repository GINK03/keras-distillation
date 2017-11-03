# Kerasを使ったDistillation(蒸留)

## Vision APIをVGG19で蒸留する
Vision APIの出力は実はタグの値を予想する問題でしかない 

出力するベクトルが任意の次元に収まっており、値の範囲を持つ場合には、特定の活性化関数で近似できる  

例えば、Vision APIはメジャーなタグに限定すれば、5000個程度のタグの予想問題であり、5000個程度であればVGG19を改良したモデルで近似できることを示す  

## 理論
去年の今頃、話題になっていたテクノロジーで、モデルのクローンが行えるとされているものである。  
Google VISION APIなどの入出力がわかれば、特定のデータセットを用意することで、何を入力したら、何が得られるかの対応が得られる 

この対応を元に、VISION APIの入出力と同等の値になるように、VGG19を学習させる  

DeepLearning界の大御所のHinton先生の論文によると、モデルはより小さいモデルで近似することができて、性能はさほど落ちないし、同等の表現力を得られるとのことです  

(重要な、仕組みはいくつかあるのですが、KL Divergenceという目的関数の作り方と、softmaxを逆算して値を出すことがポイントな気がしています)  

イメージするモデル
<div align="center">
  <img width="600px" src="https://user-images.githubusercontent.com/4949982/32262340-71c10378-bf17-11e7-9be0-f86fdf42de69.png">
</div>
<div align="center"> 図1. クローン対象のモデルをブラックボックスとして、任意のモデルで近似する </div>

## 実験環境
- pixabox.comという写真のデータセット200万枚を利用
- 特徴しては5000個の頻出する特徴を利用
- KL Divergenceではなく、Binary Cross Entropyを利用（Epoch 180時点においてこちらの方が表現力が良かった）  
- 事前に200万枚に対してGoogle Cloud Vision APIでデータセットを作成し、Distillationで使うデータセットに変換  
- AdamとSGDでAdamをとりあえず選択
- 訓練用に195万枚を利用して、テスト用に5万枚を利用した

## モデル
様々なパラメータ探索を行なったが、収束速度と学習の安定性の観点から学習済みのVGG16のモデルを改良して用いるとパフォーマンスが良かった
```python
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
```

## 結果
スナップショットして250epoch時点の結果を載せる  
まだ計算すれば性能は上がりそうである  


## 参考文献
- [ディープラーニングと著作物](https://system.jpaa.or.jp/patent/viewPdf/2741)
- [Distilling the Knowledge in a Neural Network](https://www.cs.toronto.edu/%7Ehinton/absps/distillation.pdf)




