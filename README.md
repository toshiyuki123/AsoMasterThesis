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
I run the final model on a trial basis.
``` 
  python scripts/restore_model_color_training_0.8.py
```
このとき，このように表示される．<br>
3行あり，一行目は，推論時の計測値と予測値，二行目は3つの潜在変数，三行目は前方路面画像である．<br>

In this case, it will be displayed like this.<br>

There are three lines: 
the first line is the measured and predicted values at the time of inference, 
the second line is the three latent variables, and the third line is the forward road image.
<image src="pictures/display_test.png" width=70%>


## データ収集（Data collection）

データ収集では，前方路面画像と振動情報を取得するrealsenseD435iを車載前方に搭載する．<br>
車を走行させることで前方路面画像と加速度データを取得する．<br>

For data collection, the realsenseD435i, which acquires forward road images and vibration information, is mounted in front of the vehicle.

By driving the car, we acquire the front road image and acceleration data.

走行データのrosbagである`sample.bag`は[ここ](https://drive.google.com/file/d/1lz41GKLA7QK_-HqEfRZSUWDEu1NEkdab/view?usp=sharing)からダウンロード．<br>
sample.bagにおいて，前方画像は`/camera/color/image_raw`のトピック，加速度は`/camera/accel/sample`のトピックである．<br>
これを用いて以下の3ステップ（データ前処理，学習，テスト）を説明する．

Download `sample.bag`, a rosbag of running data, from [here](https://drive.google.com/file/d/1lz41GKLA7QK_-HqEfRZSUWDEu1NEkdab/view?usp=sharing).<br>
In the `sample.bag`, the forward image is the topic of `/camera/color/image_raw` and the acceleration is the topic of `/camera/accel/sample`.<br>
The following three steps (data preprocessing, training, and testing) are described using this.<br>



## データ前処理（Data preprocessing）

このステップでは，学習モデルの入力と出力のペア（入力画像，出力振動）を生成．<br>
振動に関しては，512点の加速度情報をFFT（フーリエ変換）したものを利用．<br>

In this step, the input and output pairs (input image, output vibration) of the learning model are generated.<br>
For the vibration, we use the FFT (Fourier Transform) of the acceleration information of 512 points.<br>

`scripts/csv_saver.py`でrosbagを起動することで，時間と加速度の情報を`csv/acc.csv`に保存.<br>
Save time and acceleration information in `csv/acc.csv` by running rosbag in `scripts/csv_saver.py`.
``` 
  python scripts/csv_saver.py
  rosbag play sample.bag
```

`scripts/csv2data2.py`でrosbagを起動させることで，csvから`data/img`下に画像，`data/spec`下にスペクトログラムを保存.
Save images from csv under `data/img` and spectrograms under `data/spec` by rosbag in `scripts/csv2data2.py`.

``` 
  python scripts/csv2data2.py
  rosbag play sample.bag
```

## 学習（Train）
必要なものをインストール．<br>
そして，jupyter notebookで編集して学習．<br>

Install what you need.
Learning by editing with jupyter notebook.
```
pip install tensorflow
```
``` 
  jupyter notebook jupyter/Learning/111320-1Channel_3LatentVariables_training_for_0.8.ipynb 
```


## テスト（Test）
保存したモデルを用いてテスト．<br>
下のスクリプトの`20201119_22:40.model`を自分のモデルに置換．<br>

Test with the saved model.<br>
Replace `20201119_22:40.model` in the script below with your own model.<br>
``` 
  python scripts/restore_model_color_training_0.8.py
```
