import numpy as np

def true_function(x):
    y = np.sin(np.pi * x * 0.8) * 10
    return y

assert true_function(0) == 0
print("The unit test was successful.")

import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 100)
y = true_function(x)

plt.plot(x, y, label='y = sin(pi * x * 0.8) * 10')
plt.legend()
plt.savefig('Week3/ex1.1.png')