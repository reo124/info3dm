import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import japanize_matplotlib

# ex1.1
def true_function(x):
    y = np.sin(np.pi * x * 0.8) * 10
    return y

assert true_function(0) == 0
print("The unit test was successful.")

x = np.linspace(-1, 1, 100)
y = true_function(x)
plt.plot(x, y, label='y = sin(pi * x * 0.8) * 10')
plt.legend()
plt.savefig('ex1.1.png')

# ex1.2
np.random.seed(0)
x_obs = np.random.uniform(-1, 1, 20)

y_true = true_function(x_obs)

df = pd.DataFrame({"観測点" : x_obs, "真値" : y_true})

plt.scatter(x_obs, y_true, label='観測点', s=100)
plt.legend()
plt.savefig('ex1.2.png')

# ex1.3
noise = np.random.normal(0.0, np.sqrt(2.0), 20) / 2
obs_val = y_true + noise
df["観測値"] = obs_val

plt.scatter(x_obs, obs_val, label='観測値', marker='x', color='red')
plt.legend()
plt.savefig('ex1.3.png')