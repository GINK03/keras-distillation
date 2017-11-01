# Kerasを使ったDistillation(蒸留)

## Vision APIをVGG19で蒸留する
Vision APIの出力は実はタグの値を予想する問題でしかない 

出力するベクトルが任意の次元に収まっており、値の範囲を持つ場合には、特定の活性化関数で近似できる  

例えば、Vision APIはメジャーなタグに限定すれば、5000個程度のタグの予想問題であり、5000個程度であればVGG19を改良したモデルで近似できることを示す  

## 理論
去年の今頃、話題になっていたテクノロジーで、モデルのクローンが行えるとされているものである。  
Google VISION APIなどの入出力がわかれば、特定のデータセットを用意することで、何を入力したら、何が得られるかの対応が得られる 

この対応を元に、VISION APIの入出力と同等の値になるように、VGG19を学習させる  



