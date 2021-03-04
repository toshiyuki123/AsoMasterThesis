# AsoMasterThesis

自分の研究では画像から中間層を介して振動を出力するような学習モデルを生成する．<br>
そして，その中間層を用いることで路面を分類しようとするものである．<br>

研究をする際に以下の4ステップが必要となる．

* データ収集
* データ前処理
* 学習
* テスト

それぞれのステップについて説明する．


## データ収集

データ収集では，前方路面画像と振動情報を取得するrealsenseD435iを車載前方に搭載する．<br>
車を走行させることで前方路面画像と加速度データを取得する．

走行データのrosbagであるsample.bagは[ここ](https://drive.google.com/file/d/1lz41GKLA7QK_-HqEfRZSUWDEu1NEkdab/view?usp=sharing)からダウンロード．<br>
sample.bagにおいて，前方画像は，”/camera/color/image_raw”のトピック，加速度は，”/camera/accel/sample”のトピックである．<br>
これを用いて以下の3ステップ（データ前処理，学習，テスト）を説明する．

## データ前処理

このステップでは，学習モデルの入力と出力のペア（入力画像，出力振動）を生成する．<br>
振動に関しては，512点の加速度情報をFFT（フーリエ変換）したものを用いる．

csv_saver.pyでrosbagを起動することで，時間と加速度の情報をcsv/acc.csvに保存
``` 
  python csv_saver.py
  rosbag play sample.bag
```

csv2data.pyでrosbagを起動させることで，csvからdata/img下に画像，data/spec下にスペクトログラムを保存
``` 
  python csv2data.py
  rosbag play sample.bag
```

## 学習

jupyter notebookで学習（.ipynb）


## テスト
保存したものを用いてテストをする．
例えば，〜を用いたものだと次のように動かす．
``` 
  python test.py
```
