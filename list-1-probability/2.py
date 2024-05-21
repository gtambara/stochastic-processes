import numpy as np
from matplotlib import pyplot as plt

x = np.random.uniform(0, 1, 5000)
y = - np.log(x)

counts, bin_edges = np.histogram(y, bins=100, density=True)
plt.step(bin_edges[:-1], counts, where='mid',alpha=0.6, color='green', label='Y1=-ln(x)')

Y = np.random.exponential(1, size=5000)

def pdf_y(y):
    return np.exp(-y) if y >= 0 else 0

pdf_values = np.vectorize(pdf_y)(Y)

plt.title("PDF of Y=-ln(x) for x~U(0,1)")
counts2, bin_edges2 = np.histogram(Y, bins=100, density=True)
plt.step(bin_edges2[:-1], counts2, where='mid',alpha=0.6, color='blue', label='Y2=exp(-y)')
plt.suptitle(
          "mean of y1: " + "{:.2f}".format(np.mean(y)) + "\n" +
          "variance of y1: " + "{:.2f}".format(np.var(y)) + "\n" +
          "mean of y2: " + "{:.2f}".format(np.mean(Y)) + "\n" +
          "variance of y: " + "{:.2f}".format(np.var(Y))
          , x=0.13 , y=0.85, fontsize=10, color='gray', ha = 'left')

plt.legend()
plt.show()