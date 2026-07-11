import numpy as np
import datasets
import regression

X, Y = datasets.load_nonlinear_example1()
ex_X = datasets.polynomial3_features(X)
model = regression.LinearRegression()
model.fit(ex_X, Y)

samples = np.arange(0, 4, 0.1)
x_samples = np.c_[np.ones_like(samples), samples]
ex_x_samples = datasets.polynomial3_features(x_samples)

import matplotlib.pyplot as plt
plt.scatter(X[:, 1], Y)
plt.plot(samples, model.predict(ex_x_samples))
plt.show()