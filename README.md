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

## 実験環境
- PEXELという写真のデータセット90万枚を利用
- 特徴しては5000個の頻出する特徴を利用
- KL Divergenceではなく、Binary Cross Entropyを利用（Epoch 180時点においてこちらの方が表現力が良かった）  
- 事前に90万枚に対してGoogle Cloud Vision APIでデータセットを作成し、Distillationで使うデータセットに変換






