# Kerasを使ったGoogle VisionサービスのDistillation(蒸留)

## Vision APIをVGGで蒸留する
Vision APIの出力は実はタグの値を予想する問題でしかない 

出力するベクトルが任意の次元に収まっており、値の範囲を持つ場合には、特定の活性化関数で近似できる  

例えば、Vision APIはメジャーなタグに限定すれば、5000個程度のタグの予想問題であり、5000個程度であればVGGを改良したモデルで近似できることを示す  

## 理論
去年の今頃、話題になっていたテクノロジーで、モデルのクローンが行えるとされているものである。  
Google VISION APIなどの入出力がわかれば、特定のデータセットを用意することで、何を入力したら、何が得られるかの対応が得られる 

この対応を元に、VISION APIの入出力と同等の値になるように、VGG19を学習させる  

DeepLearning界の大御所のHinton先生の論文によると、モデルはより小さいモデルで近似することができて、性能はさほど落ちないし、同等の表現力を得られるとのことです[2]  

(蒸留を行う上で重要な、仕組みはいくつかあるのですが、KL Divergenceという目的関数の作り方と、softmaxを逆算して値を出すことがポイントな気がしています）  

イメージするモデル
<div align="center">
  <img width="600px" src="https://user-images.githubusercontent.com/4949982/32262340-71c10378-bf17-11e7-9be0-f86fdf42de69.png">
</div>
<div align="center"> 図1. クローン対象のモデルをブラックボックスとして、任意のモデルで近似する </div>

論文などによると、人間が真偽値を0,1で与えるのではなく、機械学習のモデルの出力値である 0.0 ~ 1.0までの連続値を与えることで、蒸留の対象もととなるモデルの癖や特徴量の把握の仕方まで仔細にクローンできるので効率的なのだそうです　　

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

<div align="center">
  <img width="600px" src="https://user-images.githubusercontent.com/4949982/32364114-5ea366aa-c0b6-11e7-8bdd-47ccd34c4357.png">
</div>
<div align="center"> 図2. 雪山の山脈 </div>

<div align="center">
  <img width="600px" src="https://user-images.githubusercontent.com/4949982/32364115-5eca4158-c0b6-11e7-839c-485a6716cfdf.png">
</div>
<div align="center"> 図3. ビーチと人と馬（馬が検出できていない）　</div>

<div align="center">
  <img width="600px" src="https://user-images.githubusercontent.com/4949982/32364116-5ef1eeec-c0b6-11e7-9164-e6531552a337.png">
</div>
<div align="center"> 図4. 荒野のライオン </div>

<div align="center">
  <img width="600px" src="https://user-images.githubusercontent.com/4949982/32412410-7f5bb390-c23a-11e7-935c-93fa5800bde9.png">
</div>
<div align="center"> 図5. アステカのデザインパターン </div>

<div align="center">
  <img width="600px" src="https://user-images.githubusercontent.com/4949982/32412616-ca63799e-c240-11e7-9780-ff4eae2a1932.png">
</div>
<div align="center"> 図6. 水と花の合成写真 </div>

## 学習時の注意点  

膨大な検証と試行錯誤を行なったのですが、KL Divを最小化するのもいいですが、Binary Cross Entropyの最小化でもどうにかなります  

また、分布を真似するというタスクの制約からか、分布を似せようとしてくるので、必然的に頻出回数が多いSitiationに一致していまします。こういう時は単純な力技でデータ増やすこと汎化させているのですが、今回は100万枚を超えるデータセットが必要で大変データ集めに苦労しました（１０万枚具体で見積もっていたら全然うまくいかなくて焦りました。。。）  

## プロジェクトのコード
[https://github.com/GINK03/keras-distillation:embed]

## 学習済みのモデル
minioの自宅サーバにおいておきます（常時起動している訳でないので、落ちてることもあります）
[models0][http://121.2.69.245:10002/minio/google-vision-distillation/keras-distillation/]

### 使用した学習データセット

[minimize.zip](http://121.2.69.245:10001/vision-distillation/minimize.zip)
[vision.zip](http://121.2.69.245:10001/vision-distillation/vision.zip)

## データセットを集める
[pixabayなどをスクレイピングしたスクレイパーが入っているgithubのプロジェクト](https://github.com/GINK03/image-free-download-scraper)です  
pixabayはデフォルトではタグ情報がロシア語であってちょっと扱いにくいのですが、これはこれで何かできそうです
(leveldbやBeautifulSoupなどの依存が必要です)  
**スクレイピング**
```console
$ python3 pixabay_downloader.py
```
**Google Visonが認識できるサイズにリサイズする**
```console
$ python3 google_vision.py --minimize
```
**GCPのGoogle Visionのキーを環境変数に反映させる(Pythonのスクリプトが認証に使用します)**
```console
export GOOGLE=AIzaSyDpuzmfCIAZPzug69****************
```
**Google Visionによりなんのタグがどの程度の確率でつくか計算し、結果をjsonで保存**
```console
$ python3 google_vision.py --scan
```
**学習に使用するデータセットを作ります**
```console
$ python3 --make_tag_index #<-タグ情報を作ります
$ python3 --make_pair #<-学習に使用するデータセットを作ります
```

## 学習
任意のデータセットを224x244にして255でノーマライズした状態Yvと、タグ情報のベクトルXvでタプルを作ります  
タプルをpickleでシリアライズしてgzipで圧縮したファイル一つが一つの画像のデータセットになります  
```python
one_data = gzip.compress( pickle.dumps( (X, y) ) )
```
任意のデータセットでこのフォーマットが成り立つものをdatasetのディレクトリに納めていただき、次のコマンドで実行します  
```console
$ python3 distillation.py --train
```

## 予想
学習に使用したフォーマットを引数に指定して、predオプションをつけることで予想できます  
```console
$ python3 distillation.py --pred dataset/${WHAT_YOU_WANT_TO_PREDICT}.pkl
...
(タグの確率値が表示されます)
```

## 蒸留における法的な扱い
参考文献[1]に書いてある限りしか調査していないですが、蒸留や転移学習に関しては法的な解釈がまだ定まってないように見受けられました  

> 既存の学習済みモデルのいわゆるデータコピーではないので，直ちに複製ということはできない。ただし，部分的に利用を行ったり，その結果を模したネットワークを再構築したりといった行為にあたるので，そもそも複製にあたるか否か，元の学習済みモデルの権利者（がもし存在すれば）との間でどのような利益の調整が行われるのか，許諾の要否，元のネットワークに課されていた権利上の制約（例えば，第 47 条の 7 の例外規定におけるデータベースの著作物の複製権の許諾等），といった論点がある。これらがどのように重み付けされて考慮されるかについては，文化的な側面，及び経済的な側面から，制度設計が必要であると考えられる。

ビジネスで使用するのは避けて、当面は、学術研究に留めておいた方が良さそうです。

## ライセンス
MIT

## 参考文献
- [ディープラーニングと著作物](https://system.jpaa.or.jp/patent/viewPdf/2741)
- [Distilling the Knowledge in a Neural Network](https://www.cs.toronto.edu/%7Ehinton/absps/distillation.pdf)




