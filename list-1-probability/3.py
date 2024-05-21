import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import norm

error = []

for n in range(1, 10000, 10):
    u = np.random.uniform(-1, 1, 2 * n)
    u_pairs = u.reshape(-1, 2)

    inside_counts = 0

    for pair in u_pairs:
        if((pair[0]**2 + pair[1]**2) <= 1):
            inside_counts = inside_counts + 1

    probability_inside_circle = inside_counts/n
    estimated_pi = probability_inside_circle * 4
    error.append((np.pi - estimated_pi))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 10000, 10), error, label='π - Estimiated π')
plt.xlabel('Number of samples (n)')
plt.ylabel('Error')
plt.title('Error in π estimation')
plt.suptitle(
          "mean: " + "{:.6f}".format(np.mean(error))
          , x=0.13 , y=0.85, fontsize=10, color='gray', ha = 'left')
plt.legend()
plt.show()