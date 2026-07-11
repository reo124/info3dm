import dataset1
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import japanize_matplotlib

# ex1.7
df = dataset1.ex1_5()

# ex1.8
model = LinearRegression()

# ex1.9
df_train = df[:16]
y_train = df_train["観測値"]
x_train = df_train[["観測点"]]
model.fit(x_train, y_train)

# ex1.10
df_test = df[16:]
x_test = df_test[["観測点"]]
y_pred = model.predict(x_test)
print(y_pred)

# ex1.11
x_line = np.linspace(-1, 1, 100)
y_line = dataset1.true_function(x_line)
plt.plot(x_line, y_line, label="真の関数")
plt.scatter(df["観測点"], df["観測値"], label="観測値", marker="x", color="red")
x_for_model = pd.DataFrame({"観測点" : x_line})
y_model_line = model.predict(x_for_model)
plt.plot(x_line, y_model_line, label='線形回帰モデル')
plt.legend()
plt.savefig('ex1.10.png')
print("グラフを観察すると,構築したモデルから大幅に離れている観測値が確認できる。" \
"このことから,直線のモデルで波のような真の関数を表現しようとするとズレが生じるため,不適切である。")

# ex1.12
y_test = df_test["観測値"]
mae = mean_absolute_error(y_test, y_pred)
print(mae)
print("MAEは約1.5であり, +10から-10の変動幅に対して一見すると許容範囲の誤差に思える。しかし,グラフから明らかなようにこの誤差はランダムなノイズによるものではなく,直線モデルでは波の形を表現しきれないという構造的な欠陥から生じている。したがって, MAEの数値だけを見て精度が良いと判断するのは危険であり,より複雑な曲線を描けるモデルへ変更する必要があると言える。")

# ex1.13
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# ex1.14
with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

print(loaded_model.predict(x_test))

plt.figure()
plt.plot(x_line, y_line, label="真の関数")
plt.scatter(df["観測点"], df["観測値"], label="観測値", marker="x", color="red")
y_loaded_model_line = loaded_model.predict(x_for_model)
plt.plot(x_line, y_loaded_model_line, label='線形回帰モデル(復元)')
plt.legend()
plt.savefig('ex1.14.png')

y_pred_loaded = loaded_model.predict(x_test)
mae_loaded = mean_absolute_error(y_test, y_pred_loaded)
print("復元モデルのMAE:", mae_loaded)