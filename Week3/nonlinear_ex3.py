
import numpy as np
import datasets
import regression
import matplotlib.pyplot as plt

X, Y = datasets.load_nonlinear_example1()
ex_X = datasets.polynomial3_features(X)

alphas = [0.0, 0.1, 0.5, 1.0, 10.0]
for alpha in alphas:
    model = regression.RidgeRegression(alpha=alpha)
    model.fit(ex_X, Y)

    samples = np.linspace(0, 4, 100)
    x_samples = np.c_[np.ones_like(samples), samples]
    ex_x_samples = datasets.polynomial3_features(x_samples)

    plt.plot(samples, model.predict(ex_x_samples), label=f'alpha={alpha}')
plt.scatter(X[:, 1], Y)
plt.legend()
plt.xlim(-1, 5)
plt.ylim(-6, 8)
plt.show()