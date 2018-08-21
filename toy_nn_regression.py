# http://peterroelants.github.io/posts/neural_network_implementation_part01/

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Squared error loss for least squared regression
def squared_loss(preds, truth):
    return np.sum((truth - preds) ** 2)

# Linear least squares regression
def ls_reg_nn_predict(samples, weight, bias):
    return bias + samples * weight

# Error gradient wrt weight for each sample
def weight_grad(samples, preds, truth):
    return 2 * np.sum(samples * (preds - truth))

# Same for bias but there is no depdency on any sample
def bias_grad(preds, truth):
    return 2 * np.sum(preds - truth)

# Update for the weight according to the error gradient
def weight_update(samples, preds, truth, learning_rate):
    return learning_rate * weight_grad(samples, preds, truth)

# Update for the bias according to the error gradient
def bias_update(preds, truth, learning_rate):
    return learning_rate * bias_grad(preds, truth)


# Generate the dataset (1d regression)
n_samples = 50
noise_level = 0.2
x = np.random.uniform(0, 1, n_samples)
true_w, true_b = 2, 3
t = true_b + true_w * x + np.random.normal(0, noise_level, n_samples)

# Plot the dataset
plt.figure(1)
plt.plot(x, t, 'x')
plt.plot((0, 1), (true_b, true_w + true_b))
# plt.show()

# Plot the loss by weight
grid_size = 20
weight_grid = np.linspace(-2, 8, grid_size)
bias_grid = np.linspace(0, 8, grid_size)
pred_grid = [[ls_reg_nn_predict(x, w, b) for w in weight_grid] for b in bias_grid]
loss_grid = np.array([squared_loss(p, t) for preds_b in pred_grid for p in preds_b])
loss_grid = loss_grid.reshape((grid_size, grid_size))
plt.figure(2)
# plt.pcolormesh(bias_grid, weight_grid, loss_grid)
plt.contourf(bias_grid, weight_grid, loss_grid, 30, cmap=cm.viridis)
plt.colorbar()
# plt.show()

# Training
weight = np.random.uniform(0, 1)
bias = np.random.uniform(0, 1)
n_epochs = 10
learning_rate = 0.01

weights = np.ndarray((n_epochs, 1))
biases = np.ndarray((n_epochs, 1))
losses = np.ndarray((n_epochs, 1))
for i in range(n_epochs):
    preds = ls_reg_nn_predict(x, weight, bias)
    loss = squared_loss(preds, t)
    weights[i] = weight
    biases[i] = bias
    losses[i] = loss
    print(weight, bias, loss)

    # w_grad = weight_grad(x, preds, t)
    w_upd = weight_update(x, preds, t, learning_rate)
    weight -= w_upd

    # b_grad = bias_grad(preds, t)
    b_upd = bias_update(preds, t, learning_rate)
    bias -= b_upd
    plt.text(bias, weight, str(i))
# plt.scatter(biases, weights, marker=".")
# plt.show()

plt.figure(3)
plt.plot(x, t, 'x')
plt.plot((0, 1), (3, 5))
plt.plot((0, 1), bias + (0, weight))
plt.show()
