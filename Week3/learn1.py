import dataset1
from sklearn.linear_model import LinearRegression

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