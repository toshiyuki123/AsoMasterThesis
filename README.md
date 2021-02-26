# AsoMasterThesis

自分の研究では画像から中間層を介して振動を出力するような学習モデルを生成する．
そして，その中間層を用いることで路面を分類しようとするものである．

End-to-End自動運転で必要なプログラム群を整理しておく
End-to-End自動運転では以下のステップが必要である。

* データ収集
* データ生成
* 学習
* テスト

以下ではそれぞれのステップについて説明する

## データ収集

データ収集では実際に移動体をjoystickによって走らせ、画像・点群を集める。

まずAutowareを起動する

``` 
  cd Autoware/ros
  ./run
```


NUIVの場合はssm, sh_spur, i-Cart-middleの場合はyp-spurによってロボットを走らせる。
自動起動のためのスクリプトとしてNUIVの場合は`AutoKerberos.sh`, i-Cart-middleの場合は`YPKerberos.sh`を起動する。


``` NUIV
  bash AutoKerberos.sh
```

``` i-Cart-middle
  bash YPKerberos.sh
```

収集のときはrvizにより画像・点群が出ているかを確認した後、画像・点群・姿勢情報(`/tf`または`/ypspur_ros/odom`)を以下で収集する。
(デフォルトの保存先はSamsung_T31なので要設定)

``` 
  roslaunch rosbag.lauch
```

収集後は取れたrosbagを確認して、点群、画像、姿勢情報が収集できているか確認する。

``` 
  rosbag  info sample.bag
```


## データ生成

データ生成では収集したデータをrosbagから画像(jpg), 点群(jpg), 教師データ(csv)に展開する。


まず`data_extraction`によりrosbagから画像(jpg), 点群(pcd), TFデータ(csv)を切り出す。


次に`data_argmentation`により切り出したデータを学習できるように整形+データ拡張を行う。

両方は対象となるrosbagをあるフォルダに格納した後、'genetaion_list.sh'にフォルダへのパスを追記することで実行できる。

``` 
  bash generation_lish.sh
```

実行する際は1.label3dをビルド(CMakeList.txtのパスを書き換え), 2. generation_listを書き換え, 3.実行。

実行時はローカルに`data_extraction`の結果が生成され、その後outputで指定したフォルダに`data_argmentation`の結果が生成される


サーバーで学習する場合はサーバーにデータをアップロードする。このとき、サーバーのローカルにデータを置いておくと学習が早くなる。

## 学習

学習は時間がかかるため、サーバー内で行うのが良い。

サーバーでの学習のファイルは`queue`, ローカルでの学習は`CNN`にまとめられている。
学習時は`queue/train-gpuall...`内の学習対象となるディレクトリを書き換えて、サーバーに投げる

``` 
  ssh user@192.168.1.55
  ディレクトリに移動
  sbatch -w million3 rungpu.sh 
```


## 走行

実際に走行を行う。
Autowareを起動して、NUIVであれば、Cyclops.sh, i-Cart-MiddleであればYPMaster.shを起動して準備をする。


``` NUIV
  bash Cyclops.sh
```

``` i-Cart-middle
  bash YPMaster.sh
```

rvizを確認しカメラと点群が起動できていることを確認したら、モデルを立ち上げて走行を開始する。
モデルによって立ち上げスクリプトは異なる。

``` サンプル
  python tensorflow_in_ros_pcd_2019.py
```
