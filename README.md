# Preferred Networks インターン選考2019 コーディング課題 機械学習・数理分野(長谷川太河)

課題のコードは全て、**src**ディレクトリのもとにある。

課題用のデータセットはsrcディレクトリと同じ階層にある**datasets**という名のディレクトリの中に保存されていると仮定する。

### 課題1 (Question1.py)

**Question1.py**には集約ステップを数回繰り返し、その後、特徴ベクトルを足し合わせるアルゴリズムが書かれている。コードを実行すると、集約回数2回、パラメーターは課題3で与えられた値で特徴ベクトルと隣接行列は以下の時の特徴ベクトルhを出力する。

```
#Adjacency matrix for test
G=make_symmetric_matrix(np.array([[1,0],[2,1],[3,2],[3,4],[1,3],[4,2],[4,5],[6,4],[6,3],[6,5],[7,2],[7,0],[7,3],[8,0],[8,5],[8,6],[9,2],[9,8]]),(10,10))
#feature vector when D=4
X=np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]])
```

### 課題2 (Question2.py)

**Question2.py**にはGNNのforward stepを行うGNNクラスと数値微分により勾配法を行う関数が書かれている。コードを実行すると、集約回数5回、パラメーターは課題3で与えられている値、正解ラベルは0、隣接行列は課題1と同じものの時の損失と予測ラベルをまず出力する。また勾配法を20001回実行し、1000回ごとに、その時の損失と予測ラベルも出力する。

### 課題3 (prepare_data.py,Question3.py)

**Question3.py**にはStochastic gradient descent(SGD)とmomentum stochastic gradient descentが実装されている。**Question3.py**を実行する前に**prepare_data.py**をまず実行する。**prepare_data.py**は課題3、課題4で用いるデータセットの前処理が書かれている。4/5は訓練データ、1/5は検証用データとして分割する。加工後のデータは**datasets**ディレクトリに保存される。その後、**Question3.py**を実行すると、バッチサイズ20、epoch数100、その他のパラメーターは課題３で与えられているものを利用し、SGDとMomentum SGDによるback propagationが行われる。1epochごとの訓練データでの損失と精度、検証データでの損失と精度とこれらの推移を示したグラフを出力する。グラフは**img**ディレクトリに保存される。

### 課題4 (Question4.py)

**Question4.py**ではAdamを用いて最適化がなされる。1epochごとの訓練データでの損失と精度、検証データでの損失と精度とこれらの推移を示したグラフに加え、テストデータに対する予測ラベルが記された**prediction.txt**が出力される。グラフは課題3と同じく**img**ディレクトリに保存される。
