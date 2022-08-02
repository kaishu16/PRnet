# PRnet

参考論文
[1-s2.0-S1746809420304936-main.pdf](https://github.com/kaishu16/PRnet/files/9241286/1-s2.0-S1746809420304936-main.pdf)

## データセット

![image](https://user-images.githubusercontent.com/43696731/182383553-5fed908e-c282-428a-a411-bd2ce46b183d.png)

上記画像の頬部分のバウンディングボックスを切り取った画像を用意


## model.py
以下のモデルを参考に作成

入力画像が60x128x256x3(30fpsで　2秒分のフレーム画像群)なのですが動画が25fpsだったので2秒分の50x128x256x3にしています。

- Loss: MSE loss
- optimizer: Adam

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

epoch数: 30

- 1から10 learning rate 1e-3
- 11から20 learning rate 1e-4
- 21から30 learning rate 1e-5
論文と学習率は変えていますが論文の通りの学習率でも同様の結末になりました。
の条件で学習した際、

## 問題点
- lossは減って収束に向かっているが、2桁3桁になることがずっとある。
- 予測値が一定値(79bpm周り)になってしまい、正解値に全然対応していてない(100以上や60以下のbpmがあるが全然対応できていない)。


### lossのログ　　iteration50ごとのロスになります。

```[tensor(5182.8413, device='cuda:2'), tensor(6262.5566, device='cuda:2'), tensor(5522.8892, device='cuda:2'), tensor(6890.9062, device='cuda:2'), tensor(7717.6938, device='cuda:2'), tensor(3363.6802, device='cuda:2'), tensor(4777.7656, device='cuda:2'), tensor(3283.8672, device='cuda:2'), tensor(7007.1475, device='cuda:2'), tensor(2751.6255, device='cuda:2'), tensor(1700.0010, device='cuda:2'), tensor(1744.3005, device='cuda:2'), tensor(1994.3917, device='cuda:2'), tensor(2201.1665, device='cuda:2'), tensor(959.9606, device='cuda:2'), tensor(3088.7087, device='cuda:2'), tensor(295.1588, device='cuda:2'), tensor(1836.0396, device='cuda:2'), tensor(2525.4014, device='cuda:2'), tensor(2706.8706, device='cuda:2'), tensor(697.4777, device='cuda:2'), tensor(1480.4980, device='cuda:2'), tensor(612.7181, device='cuda:2'), tensor(783.8030, device='cuda:2'), tensor(503.0187, device='cuda:2'), tensor(1007.9014, device='cuda:2'), tensor(799.1138, device='cuda:2'), tensor(1444.1748, device='cuda:2'), tensor(1937.5076, device='cuda:2'), tensor(235.0135, device='cuda:2'), tensor(749.7238, device='cuda:2'), tensor(275.9478, device='cuda:2'), tensor(2664.3455, device='cuda:2'), tensor(192.2358, device='cuda:2'), tensor(43.8801, device='cuda:2'), tensor(30.8903, device='cuda:2'), tensor(92.4805, device='cuda:2'), tensor(175.1999, device='cuda:2'), tensor(19.8730, device='cuda:2'), tensor(561.8870, device='cuda:2'), tensor(182.4036, device='cuda:2'), tensor(176.6290, device='cuda:2'), tensor(480.8354, device='cuda:2'), tensor(603.1948, device='cuda:2'), tensor(0.7505, device='cuda:2'), tensor(175.5121, device='cuda:2'), tensor(56.0588, device='cuda:2'), tensor(137.5045, device='cuda:2'), tensor(0.2383, device='cuda:2'), tensor(116.4119, device='cuda:2'), tensor(60.9374, device='cuda:2'), tensor(324.0790, device='cuda:2'), tensor(626.0236, device='cuda:2'), tensor(16.9702, device='cuda:2'), tensor(101.2970, device='cuda:2'), tensor(0.4924, device='cuda:2'), tensor(1815.3805, device='cuda:2'), tensor(2.8877, device='cuda:2'), tensor(151.0657, device='cuda:2'), tensor(61.2784, device='cuda:2'), tensor(12.2872, device='cuda:2'), tensor(16.4362, device='cuda:2'), tensor(183.0191, device='cuda:2'), tensor(194.5197, device='cuda:2'), tensor(503.6959, device='cuda:2'), tensor(25.5279, device='cuda:2'), tensor(210.0815, device='cuda:2'), tensor(302.2755, device='cuda:2'), tensor(45.0399, device='cuda:2'), tensor(53.4070, device='cuda:2'), tensor(96.9241, device='cuda:2'), tensor(145.1523, device='cuda:2'), tensor(25.3757, device='cuda:2'), tensor(46.3872, device='cuda:2'), tensor(14.4824, device='cuda:2'), tensor(188.0340, device='cuda:2'), tensor(447.3691, device='cuda:2'), tensor(53.4613, device='cuda:2'), tensor(46.1100, device='cuda:2'), tensor(9.1005, device='cuda:2'), tensor(1719.3879, device='cuda:2'), tensor(15.0537, device='cuda:2'), tensor(206.1207, device='cuda:2'), tensor(92.6572, device='cuda:2'), tensor(22.8199, device='cuda:2'), tensor(14.4694, device='cuda:2'), tensor(215.2306, device='cuda:2'), tensor(170.7116, device='cuda:2'), tensor(532.3811, device='cuda:2'), tensor(20.5931, device='cuda:2'), tensor(202.7471, device='cuda:2'), tensor(286.3257, device='cuda:2'), tensor(50.5868, device='cuda:2'), tensor(49.8451, device='cuda:2'), tensor(101.4724, device='cuda:2'), tensor(148.3996, device='cuda:2'), tensor(29.2364, device='cuda:2'), tensor(42.8707, device='cuda:2'), tensor(12.1773, device='cuda:2'), tensor(171.8541, device='cuda:2'), tensor(426.5886, device='cuda:2'), tensor(60.5362, device='cuda:2'), tensor(40.7655, device='cuda:2'), tensor(11.4188, device='cuda:2'), tensor(1710.8910, device='cuda:2'), tensor(17.5615, device='cuda:2'), tensor(213.2588, device='cuda:2'), tensor(94.3543, device='cuda:2'), tensor(22.7940, device='cuda:2'), tensor(14.4956, device='cuda:2'), tensor(215.0226, device='cuda:2'), tensor(173.3383, device='cuda:2'), tensor(524.6998, device='cuda:2'), tensor(22.0428, device='cuda:2'), tensor(209.6329, device='cuda:2'), tensor(288.9323, device='cuda:2'), tensor(49.6455, device='cuda:2'), tensor(51.3372, device='cuda:2'), tensor(101.0885, device='cuda:2'), tensor(148.7689, device='cuda:2'), tensor(29.5633, device='cuda:2'), tensor(42.5551, device='cuda:2'), tensor(11.8122, device='cuda:2'), tensor(167.0939, device='cuda:2'), tensor(420.8531, device='cuda:2'), tensor(62.6913, device='cuda:2'), tensor(39.3872, device='cuda:2'), tensor(12.0779, device='cuda:2'), tensor(1709.1501, device='cuda:2'), tensor(18.2862, device='cuda:2'), tensor(215.0966, device='cuda:2'), tensor(94.3450, device='cuda:2'), tensor(22.5213, device='cuda:2'), tensor(14.5218, device='cuda:2'), tensor(214.3496, device='cuda:2'), tensor(174.8147, device='cuda:2'), tensor(503.8008, device='cuda:2'), tensor(24.7160, device='cuda:2'), tensor(213.0555, device='cuda:2'), tensor(284.0269, device='cuda:2'), tensor(51.3890, device='cuda:2'), tensor(50.1129, device='cuda:2'), tensor(102.4340, device='cuda:2'), tensor(150.2609, device='cuda:2'), tensor(30.4964, device='cuda:2'), tensor(41.7478, device='cuda:2'), tensor(11.3571, device='cuda:2'), tensor(164.1171, device='cuda:2'), tensor(417.1151, device='cuda:2'), tensor(63.9956, device='cuda:2'), tensor(38.5408, device='cuda:2'), tensor(12.5014, device='cuda:2'), tensor(1707.7317, device='cuda:2'), tensor(17.7891, device='cuda:2'), tensor(216.3661, device='cuda:2'), tensor(89.2582, device='cuda:2'), tensor(19.5740, device='cuda:2'), tensor(14.1876, device='cuda:2'), tensor(211.5698, device='cuda:2'), tensor(183.0532, device='cuda:2'), tensor(498.0308, device='cuda:2'), tensor(25.3488, device='cuda:2'), tensor(236.7296, device='cuda:2'), tensor(286.8500, device='cuda:2'), tensor(48.9865, device='cuda:2'), tensor(49.8558, device='cuda:2'), tensor(106.3670, device='cuda:2'), tensor(154.8591, device='cuda:2'), tensor(31.4190, device='cuda:2'), tensor(42.3817, device='cuda:2'), tensor(11.5702, device='cuda:2'), tensor(159.5681, device='cuda:2'), tensor(413.8533, device='cuda:2'), tensor(66.2967, device='cuda:2'), tensor(37.8866, device='cuda:2'), tensor(13.1335, device='cuda:2'), tensor(1693.3698, device='cuda:2'), tensor(19.4419, device='cuda:2'), tensor(220.5121, device='cuda:2'), tensor(92.8246, device='cuda:2'), tensor(21.1458, device='cuda:2'), tensor(13.6076, device='cuda:2'), tensor(214.7697, device='cuda:2'), tensor(178.2057, device='cuda:2'), tensor(510.2383, device='cuda:2'), tensor(23.2121, device='cuda:2'), tensor(223.6765, device='cuda:2'), tensor(289.7240, device='cuda:2'), tensor(50.1920, device='cuda:2'), tensor(51.8631, device='cuda:2'), tensor(101.0222, device='cuda:2'), tensor(153.3111, device='cuda:2'), tensor(30.0698, device='cuda:2'), tensor(42.5786, device='cuda:2'), tensor(11.2448, device='cuda:2'), tensor(161.1020, device='cuda:2'), tensor(412.7238, device='cuda:2'), tensor(65.5433, device='cuda:2'), tensor(38.0147, device='cuda:2'), tensor(13.2446, device='cuda:2'), tensor(1701.4581, device='cuda:2'), tensor(19.1461, device='cuda:2'), tensor(218.8712, device='cuda:2'), tensor(93.8435, device='cuda:2'), tensor(22.6931, device='cuda:2'), tensor(13.8167, device='cuda:2'), tensor(214.4589, device='cuda:2'), tensor(178.5610, device='cuda:2'), tensor(514.4449, device='cuda:2'), tensor(24.6120, device='cuda:2'), tensor(221.9869, device='cuda:2'), tensor(292.2736, device='cuda:2'), tensor(49.2713, device='cuda:2'), tensor(53.4673, device='cuda:2'), tensor(99.8736, device='cuda:2'), tensor(152.3439, device='cuda:2'), tensor(30.3729, device='cuda:2'), tensor(41.9356, device='cuda:2'), tensor(11.6172, device='cuda:2'), tensor(162.1508, device='cuda:2'), tensor(414.1664, device='cuda:2'), tensor(65.0428, device='cuda:2'), tensor(37.8588, device='cuda:2'), tensor(12.4984, device='cuda:2'), tensor(1712.1837, device='cuda:2'), tensor(19.3072, device='cuda:2'), tensor(216.5467, device='cuda:2'), tensor(93.9825, device='cuda:2'), tensor(21.7384, device='cuda:2'), tensor(14.3015, device='cuda:2'), tensor(212.9705, device='cuda:2'), tensor(175.3863, device='cuda:2'), tensor(517.1624, device='cuda:2'), tensor(22.7181, device='cuda:2'), tensor(219.5620, device='cuda:2'), tensor(292.0198, device='cuda:2'), tensor(50.2227, device='cuda:2'), tensor(52.0409, device='cuda:2'), tensor(102.2172, device='cuda:2'), tensor(151.2383, device='cuda:2'), tensor(32.0660, device='cuda:2'), tensor(40.1906, device='cuda:2'), tensor(13.4761, device='cuda:2'), tensor(213.2759, device='cuda:2'), tensor(482.8911, device='cuda:2'), tensor(39.3825, device='cuda:2'), tensor(61.9150, device='cuda:2'), tensor(3.5449, device='cuda:2'), tensor(1754.7782, device='cuda:2'), tensor(6.6914, device='cuda:2'), tensor(170.6142, device='cuda:2'), tensor(80.9107, device='cuda:2'), tensor(21.0533, device='cuda:2'), tensor(14.0631, device='cuda:2'), tensor(212.1717, device='cuda:2'), tensor(156.8741, device='cuda:2'), tensor(582.5063, device='cuda:2'), tensor(11.0527, device='cuda:2'), tensor(151.7557, device='cuda:2'), tensor(255.0147, device='cuda:2'), tensor(65.9597, device='cuda:2'), tensor(32.5404, device='cuda:2'), tensor(114.3169, device='cuda:2'), tensor(151.7352, device='cuda:2'), tensor(32.7566, device='cuda:2'), tensor(39.6865, device='cuda:2'), tensor(13.1004, device='cuda:2'), tensor(211.5341, device='cuda:2'), tensor(480.9783, device='cuda:2'), tensor(40.0563, device='cuda:2'), tensor(60.8119, device='cuda:2'), tensor(3.7377, device='cuda:2'), tensor(1753.4075, device='cuda:2'), tensor(6.7369, device='cuda:2'), tensor(172.4012, device='cuda:2'), tensor(81.8972, device='cuda:2'), tensor(21.4591, device='cuda:2'), tensor(14.0661, device='cuda:2'), tensor(213.5084, device='cuda:2'), tensor(155.6347, device='cuda:2'), tensor(585.4646, device='cuda:2'), tensor(10.7689, device='cuda:2'), tensor(150.6277, device='cuda:2'), tensor(252.6355, device='cuda:2'), tensor(66.5906, device='cuda:2'), tensor(31.9412, device='cuda:2'), tensor(115.0584, device='cuda:2'), tensor(151.8434, device='cuda:2'), tensor(32.9344, device='cuda:2'), tensor(39.7696, device='cuda:2'), tensor(12.8949, device='cuda:2'), tensor(210.5264, device='cuda:2'), tensor(478.0032, device='cuda:2'), tensor(40.6036, device='cuda:2'), tensor(60.3051, device='cuda:2'), tensor(3.8541, device='cuda:2'), tensor(1752.3401, device='cuda:2'), tensor(6.8967, device='cuda:2'), tensor(172.6976, device='cuda:2'), tensor(82.3677, device='cuda:2'), tensor(21.7512, device='cuda:2'), tensor(14.0603, device='cuda:2'), tensor(214.2547, device='cuda:2'), tensor(155.2294, device='cuda:2'), tensor(585.8416, device='cuda:2'), tensor(10.7276, device='cuda:2'), tensor(150.8798, device='cuda:2'), tensor(251.9826, device='cuda:2'), tensor(66.9924, device='cuda:2'), tensor(31.9346, device='cuda:2'), tensor(115.3055, device='cuda:2'), tensor(151.4806, device='cuda:2'), tensor(33.1173, device='cuda:2'), tensor(39.5754, device='cuda:2'), tensor(12.7768, device='cuda:2'), tensor(209.9709, device='cuda:2'), tensor(477.0513, device='cuda:2'), tensor(40.7303, device='cuda:2'), tensor(60.1140, device='cuda:2'), tensor(3.9216, device='cuda:2'), tensor(1751.2683, device='cuda:2'), tensor(7.0810, device='cuda:2'), tensor(172.9645, device='cuda:2'), tensor(82.4550, device='cuda:2'), tensor(21.7984, device='cuda:2'), tensor(14.0500, device='cuda:2'), tensor(214.6756, device='cuda:2'), tensor(154.9696, device='cuda:2'), tensor(585.4168, device='cuda:2'), tensor(10.5601, device='cuda:2')]　```


### テストデータの結果

testデータをモデルに通した際の予測値と正解値を示しています。

![スクリーンショット 2022-08-02 15 18 53](https://user-images.githubusercontent.com/43696731/182352956-0516c5c6-d6cf-4693-85c0-888b097cc86b.png)


