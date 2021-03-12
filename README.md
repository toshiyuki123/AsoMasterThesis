# AsoMasterThesis

自分の研究では画像から中間層を介して振動を出力するような学習モデルを生成する．<br>
そして，その中間層を用いることで路面を分類しようとするものである．<br>

以下の5ステップを説明する．

* 試しに最終形を動かす
* データ収集
* データ前処理
* 学習
* テスト

それぞれのステップについて説明する．

In my research, I generate a learning model that outputs vibrations from images through an intermediate layer.
Then, I classify the road surface by using the intermediate layer.

The following 5 steps will be explained.

* Run the final model on a trial basis
* Data collection
* Data preprocessing
* Train
* Test

Each of these steps will be explained.

## 試しに最終形を動かす（Run the final model on a trial basis）

試しに最終形を動かしてみる．<br>
``` 
  python scripts/restore_model_color_training_0.8.py
```
このとき，このように表示される．<br>
3行あり，一行目は，推論時の計測値と予測値，二行目は3つの潜在変数，三行目は前方路面画像である．
<image src="pictures/display_test.png" width=70%>


## データ収集（Data collection）

データ収集では，前方路面画像と振動情報を取得するrealsenseD435iを車載前方に搭載する．<br>
車を走行させることで前方路面画像と加速度データを取得する．

走行データのrosbagである`sample.bag`は[ここ](https://drive.google.com/file/d/1lz41GKLA7QK_-HqEfRZSUWDEu1NEkdab/view?usp=sharing)からダウンロード．<br>
sample.bagにおいて，前方画像は`/camera/color/image_raw`のトピック，加速度は`/camera/accel/sample`のトピックである．<br>
これを用いて以下の3ステップ（データ前処理，学習，テスト）を説明する．

## データ前処理（Data preprocessing）

このステップでは，学習モデルの入力と出力のペア（入力画像，出力振動）を生成．<br>
振動に関しては，512点の加速度情報をFFT（フーリエ変換）したものを利用．

`scripts/csv_saver.py`でrosbagを起動することで，時間と加速度の情報を`csv/acc.csv`に保存
``` 
  python scripts/csv_saver.py
  rosbag play sample.bag
```

`scripts/csv2data2.py`でrosbagを起動させることで，csvから`data/img`下に画像，`data/spec`下にスペクトログラムを保存
``` 
  python scripts/csv2data2.py
  rosbag play sample.bag
```

## 学習（Train）

jupyter notebookで編集して学習．
必要なものをインストール．<br>
```
pip install tensorflow
```
``` 
  jupyter notebook jupyter/Learning/111320-1Channel_3LatentVariables_training_for_0.8.ipynb 
```


## テスト（Test）
保存したモデルを用いてテストをする．
下のスクリプトの`20201119_22:40.model`を自分のモデルに置換する．
``` 
  python scripts/restore_model_color_training_0.8.py
```
