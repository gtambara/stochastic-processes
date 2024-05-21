import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import sqrtm, polar
import random

mean = np.array([2, 0])
covariance = np.array([[1, 0.2], [0.2, 3]])

x = np.random.default_rng().multivariate_normal(mean, covariance, 5000, method='cholesky')

# 1-a)

inferior_tri = np.linalg.cholesky(covariance)
superior_tri = np.transpose(inferior_tri)
inv_sup_tri = np.linalg.inv(superior_tri)
u, p = polar(inv_sup_tri, 'left')
sqrtm_covariance = inv_sup_tri * np.transpose(u)
x_centered = x - mean
y1 = np.matmul(x_centered, sqrtm_covariance)

# 1-b)

def covar(x1, x2):
    return np.mean(x1 * x2) - np.mean(x1) * np.mean(x2)

def correlation(x1, x2):
    return covar(x1, x2)/(np.sqrt(covar(x1, x1)) * np.sqrt(covar(x2, x2)))

y2 = x_centered**2

#plotting the distributions
plt.figure(figsize=(8, 6))
plt.scatter(x[:, 0], x[:, 1], color='green', marker='x')
plt.title('Scatter plot of X Gaussian')
plt.suptitle(
          "mean of x[0]: " + "{:.2f}".format(np.mean(x[:,0])) + "\n" +
          "mean of x[1]: " + "{:.2f}".format(np.mean(x[:,1])) + "\n" +
          "variance of x[0]: " + "{:.2f}".format(np.var(x[:,0])) + "\n" +
          "variance of x[1]: " + "{:.2f}".format(np.var(x[:,1])) + '\n' +
          "covariance of x: " + "{:.2f}".format((covar(x[:,0], x[:,1]))) + '\n' +
          "correlation of σ: " + "{:.4f}".format((correlation(x[:,0], x[:,1])))
          , x=0.13 , y=0.85, fontsize=10, color='gray', ha = 'left')
plt.axis('equal')

plt.figure(figsize=(8, 6))
plt.scatter(y1[:, 0], y1[:, 1], color='blue', marker='.')
plt.title('Scatter plot of Y = P^(-1/2)(X-E{X})')
plt.suptitle(
          "mean of y1[0]: " + "{:.2f}".format(np.mean(y1[:,0])) + "\n" +
          "mean of y1[1]: " + "{:.2f}".format(np.mean(y1[:,0])) + "\n" +
          "variance of y1[0]: " + "{:.2f}".format(np.var(y1[:,0])) + "\n" +
          "variance of y1[1]: " + "{:.2f}".format(np.var(y1[:,1])) + '\n' +
          "covariance of y1: " + "{:.2f}".format(covar(y1[:,0], y1[:,1]))
          , x=0.13 , y=0.85, fontsize=10, color='gray', ha = 'left')
plt.axis('equal')

plt.figure(figsize=(8, 6))
plt.scatter(y2[:, 0], y2[:, 1], color='red', marker='.')
plt.title('Scatter plot of y=|(X-E{X})|^2')
plt.suptitle(
          "mean of y2[0]: " + "{:.2f}".format(np.mean(y2[:,0])) + "\n" +
          "mean of y2[1]: " + "{:.2f}".format((np.mean(y2[:,1]))) + "\n" +
          "variance of y2[0]: " + "{:.2f}".format(np.var(y2[:,0])) + "\n" +
          "variance of y2[1]: " + "{:.2f}".format(np.var(y2[:,1])) + '\n' +
          "covariance of y2: " + "{:.2f}".format(covar(y2[:,0], y2[:,1]))
          , x=0.13 , y=0.85, fontsize=10, color='gray', ha = 'left')
plt.axis('equal')

#plotting the PDFs of y1 and y2
hist_y1, xedges_y1, yedges_y1 = np.histogram2d(y1[:,0], y1[:,1])
hist_y2, xedges_y2, yedges_y2 = np.histogram2d(y2[:,0], y2[:,1])

xpos_y1, ypos_y1 = np.meshgrid(xedges_y1[:-1], yedges_y1[:-1])
xpos_y1 = xpos_y1.flatten('F')
ypos_y1 = ypos_y1.flatten('F')
zpos_y1 = np.zeros_like(xpos_y1)

xpos_y2, ypos_y2 = np.meshgrid(xedges_y2[:-1], yedges_y2[:-1])
xpos_y2 = xpos_y2.flatten('F')
ypos_y2 = ypos_y2.flatten('F')
zpos_y2 = np.zeros_like(xpos_y2)

dx_y1 = dy_y1 = 0.5 * np.ones_like(zpos_y1)
dz_y1 = hist_y1.flatten()
dx_y2 = dy_y2 = 0.5 * np.ones_like(zpos_y2)
dz_y2 = hist_y2.flatten()

fig = plt.figure(figsize=(14, 7))

ax1 = fig.add_subplot(121, projection='3d')
ax1.bar3d(xpos_y1, ypos_y1, zpos_y1, dx_y1, dy_y1, dz_y1, color='b')
ax1.set_title('3D Histogram for y1')
ax1.set_xlabel('Y[0] axis')
ax1.set_ylabel('Y[1] axis')
ax1.set_zlabel('Frequency')

ax2 = fig.add_subplot(122, projection='3d')
ax2.bar3d(xpos_y2, ypos_y2, zpos_y2, dx_y2, dy_y2, dz_y2, color='r')
ax2.set_title('3D Histogram for y2')
ax2.set_xlabel('Y[0] axis')
ax2.set_ylabel('Y[1] axis')
ax2.set_zlabel('Frequency')

plt.show()