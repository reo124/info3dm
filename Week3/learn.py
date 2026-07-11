import dataset1
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import japanize_matplotlib

# ex1.7
def ex1_7():
    df = dataset1.ex1_5()
    return df

# ex1.8
def ex1_8():
    model = LinearRegression()
    return model

# ex1.9
def ex1_9(df, model):
    df_train = df[:16]
    y_train = df_train["観測値"]
    x_train = df_train[["観測点"]]
    model.fit(x_train, y_train)
    return model

# ex1.10
def ex1_10(df, model):
    df_test = df[16:]
    x_test = df_test[["観測点"]]
    y_pred = model.predict(x_test)
    print("予測結果:", y_pred)
    return df_test, x_test, y_pred

# ex1.11
def ex1_11(df, model):
    x_line = np.linspace(-1, 1, 100)
    y_line = dataset1.true_function(x_line)
    
    plt.figure() 
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
def ex1_12(df_test, y_pred):
    y_test = df_test["観測値"]
    mae = mean_absolute_error(y_test, y_pred)
    print("MAE:", mae)
    print("MAEは約1.5であり, +10から-10の変動幅に対して一見すると許容範囲の誤差に思える。しかし,グラフから明らかなようにこの誤差はランダムなノイズによるものではなく,直線モデルでは波の形を表現しきれないという構造的な欠陥から生じている。したがって, MAEの数値だけを見て精度が良いと判断するのは危険であり,より複雑な曲線を描けるモデルへ変更する必要があると言える。")

# ex1.13
def ex1_13(model):
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

# ex1.14
def ex1_14(df, df_test, x_test):
    with open("model.pkl", "rb") as f:
        loaded_model = pickle.load(f)

    print("復元モデルの予測結果:", loaded_model.predict(x_test))

    x_line = np.linspace(-1, 1, 100)
    y_line = dataset1.true_function(x_line)

    plt.figure()
    plt.plot(x_line, y_line, label="真の関数")
    plt.scatter(df["観測点"], df["観測値"], label="観測値", marker="x", color="red")
    
    x_for_model = pd.DataFrame({"観測点" : x_line})
    y_loaded_model_line = loaded_model.predict(x_for_model)
    
    plt.plot(x_line, y_loaded_model_line, label='線形回帰モデル(復元)')
    plt.legend()
    plt.savefig('ex1.14.png')

    y_pred_loaded = loaded_model.predict(x_test)
    y_test = df_test["観測値"]
    mae_loaded = mean_absolute_error(y_test, y_pred_loaded)
    print("復元モデルのMAE:", mae_loaded)

if __name__ == '__main__':
    df = ex1_7()
    model = ex1_8()
    model = ex1_9(df, model)
    df_test, x_test, y_pred = ex1_10(df, model)
    ex1_11(df, model)
    ex1_12(df_test, y_pred)
    ex1_13(model)
    ex1_14(df, df_test, x_test)