import numpy as np

# ---ベクトル演算の演習---
# (1) 5個の要素を持つ列ベクトルを作成せよ。値は全て1とする。
Column_vector = np.ones((5,1))
print(Column_vector)

# (2) 1で作成した列ベクトルのうち、2番目の要素を3.14に更新せよ。なおインデックスは0から数える（0番目、1番目、2番目、、）ものとする。
Column_vector[2, 0] = 3.14
print(Column_vector)

# (3) 2で作成した列ベクトルを複製し、転置により行ベクトルに変換せよ。
Row_vector = Column_vector.copy().T
print(Row_vector)

# (4) 用意した列ベクトルと行ベクトルの内積を求めよ。
print((Row_vector @ Column_vector).item())

# (5) np.random.randを用いて、10個の要素を持つ列ベクトルを作成せよ。
print(np.random.rand(10, 1))

# ---行列演算の演習行列演算の演習---
# (6) np.random.normalを用いて、平均値10、標準偏差2の正規分布に基づく、2行5列の行列を作成せよ。
Matrix_A = np.random.normal(10, 2, (2, 5))
print(Matrix_A)

# (7) 6で作成した行列から、2列目の要素を抜き出だせ。
print(Matrix_A[:, 1:2])

# (8) 6で作成した行列から、3列目と4列目の要素を抜き出せ。
print(Matrix_A[:, 2:4])

# (9) np.random.randで5行2列の行列を用意し、6で用意した行列との積を求めよ。
Matrix_B = np.random.rand(5,2)
print(Matrix_A @ Matrix_B)