# PRnet

参考論文
[1-s2.0-S1746809420304936-main.pdf](https://github.com/kaishu16/PRnet/files/9241286/1-s2.0-S1746809420304936-main.pdf)

## データセット

![image](https://user-images.githubusercontent.com/43696731/182383553-5fed908e-c282-428a-a411-bd2ce46b183d.png)

上記画像の頬部分のバウンディングボックスを切り取った画像を用意


## model.py
以下のモデルを参考に作成

入力画像が60x128x256x3(30fpsで　2秒分のフレーム画像群)なのですが動画が25fpsだったので2秒分の50x128x256x3にしています。

### 設定値
- Loss: MSE loss
- optimizer: Adam
- epoch数: 30

![image](https://user-images.githubusercontent.com/43696731/182303896-ec3b10b1-ccda-45c4-aabb-1f00ada5b777.png)
![スクリーンショット 2022-08-02 12 59 48](https://user-images.githubusercontent.com/43696731/182303792-550c965f-dd13-4d82-bd42-7423778b156c.png)

論文に基づき
- 各Conv層の後にBatchNormalizationを設置
- 各ConvおよびLSTM層の後にLeaky ReLUを設置


## dataset.py
data.pyで振り分けたtrainとtestのデータから学習(テスト)動画像データと正解ラベル(心拍)を引っ張ってきてtensor型にしてGPUで学習できるような状態、Dataloaderの引数として代入している。

## data.py
trainとtestのフォルダにtest.pyで作成したデータを振り分ける

## test.py
学習とテストで使用するデータ(頬部分のboundingboxを切り取った50x128x256x3の画像)とラベルデータ(2秒間の心拍平均)をVIPL-HRデータセットから作成


# 現状の課題

## 問題点
- ある点を超えるとlossが減らなくなってしまっている。
- 予測値が一定値(79bpm周り)になってしまい、正解値に全然対応していてない(100以上や60以下のbpmがあるが全然対応できていない)。

## 変更点
前回lineでヘルプを求めた際kernel sizeの部分と画素値の調整に関して指摘を受けたため、kernel size部分を論文のように3x3x3の形にし、画素値に関しては正規化を行った。

## 結果　
lossグラフ
テストデータの結果 (testデータをモデルに通した際の予測値と正解値を示しています。)

1. 1つのGPUで学習させた時 (バッチサイズ: 8, 学習率 1から10: 1e-3 11から20: 1e-4 21から30: 1e-5 )

- lossグラフ
- テストデータの結果

2. 2つのGPUで学習させた時 (バッチサイズ: 16, 学習率 1から10: 2*1e-3 11から20: 2*1e-4 21から30: 2*1e-4 )

- lossグラフ
- テストデータの結果

3. 2つのGPUで学習させた時 (バッチサイズ: 16, 学習率 1から10: 2*1e-3 11から20: 2*1e-4 21から30: 2*1e-5 )

- lossグラフ
- テストデータの結果



#　知りたいこと
どこに問題点があるのか
- 学習データ
- モデルの中身
- 学習率の設定
- その他






