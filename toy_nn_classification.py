# http://peterroelants.github.io/posts/neural_network_implementation_part02/

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, colorConverter

# Log loss for classification (logistic regression)
def log_loss(preds, truth):
    return - np.sum(truth * np.log(preds) + (1 - truth) * np.log(1 - preds))

def logistic(x):
    return 1 / (1 + np.exp(-x))

# Logistic regression
def logit_classif_nn_predict_proba(samples, weights, bias):
    return logistic(bias + samples.dot(weights.T))

def logit_classif_nn_predict(samples, weights, bias):
    return np.around(logit_classif_nn_predict_proba(samples, weights, bias))

# Error gradient wrt weight for each sample
def weight_grad(samples, preds, truth):
    # print(samples)
    # print(preds - truth)
    return (preds - truth).T.dot(samples)

# Same for bias but there is no depdency on any sample
def bias_grad(preds, truth):
    return np.sum(preds - truth)

# Update for the weight according to the error gradient
def weight_update(samples, preds, truth, learning_rate):
    return learning_rate * weight_grad(samples, preds, truth)

# Update for the bias according to the error gradient
def bias_update(preds, truth, learning_rate):
    return learning_rate * bias_grad(preds, truth)


# Generate the dataset (2d classification)
n_samples_per_class = 25
noise_level = 0.5
class_1_mean =[0, 0]
class_2_mean = [1, 1]
x1 = np.random.randn(n_samples_per_class, 2) * noise_level + class_1_mean
x2 = np.random.randn(n_samples_per_class, 2) * noise_level + class_2_mean

X = np.vstack((x1, x2))
t = np.hstack((np.zeros(n_samples_per_class), np.ones(n_samples_per_class)))

# Plot the dataset
plt.figure(1)
plt.plot(x1[:, 0], x1[:, 1], 'o', c="red")
plt.plot(x2[:, 0], x2[:, 1], 'o', c="blue")
# plt.show()

# Plot the loss by weight (fixed bias = 0)
grid_size = 20
w1_grid = np.linspace(-2, 5, grid_size)
w2_grid = np.linspace(-2, 5, grid_size)
pred_grid = [[logit_classif_nn_predict_proba(X, np.array([w1, w2]), 0) for w2 in w2_grid] for w1 in w1_grid]
loss_grid = np.array([log_loss(p, t) for preds_w in pred_grid for p in preds_w])
loss_grid = loss_grid.reshape((grid_size, grid_size))
plt.figure(2)
# plt.pcolormesh(w1_grid, w1_grid, loss_grid)
plt.contourf(w1_grid, w2_grid, loss_grid, 30, cmap=cm.viridis)
plt.colorbar()
# plt.show()

# Training
weights = np.array([np.random.uniform(0, 1), np.random.uniform(0, 1)])
bias = np.random.uniform(0, 1)
n_epochs = 50
learning_rate = 0.05

losses = np.ndarray(n_epochs)
for i in range(n_epochs):
    preds = logit_classif_nn_predict_proba(X, weights, bias)
    loss = log_loss(preds, t)
    losses[i] = loss
    print(weights, bias, loss)

    # w_grad = weight_grad(X, preds, t)
    w_upd = weight_update(X, preds, t, learning_rate)
    weights -= w_upd

    # b_grad = bias_grad(preds, t)
    b_upd = bias_update(preds, t, learning_rate)
    bias -= b_upd

plt.figure(3)
plt.plot(losses)
# plt.show()

nb_of_xs = 200
xs1 = np.linspace(-4, 4, num=nb_of_xs)
xs2 = np.linspace(-4, 4, num=nb_of_xs)
xx, yy = np.meshgrid(xs1, xs2) # create the grid
# Initialize and fill the classification plane
classification_plane = np.zeros((nb_of_xs, nb_of_xs))
for i in range(nb_of_xs):
    for j in range(nb_of_xs):
        classification_plane[i, j] = logit_classif_nn_predict(np.asmatrix([xx[i, j], yy[i, j]]), weights, bias)
# Create a color map to show the classification colors of each grid point
cmap = ListedColormap([
        colorConverter.to_rgba('r', alpha=0.30),
        colorConverter.to_rgba('b', alpha=0.30)])

# Plot the classification plane with decision boundary and input samples
plt.figure(4)
plt.contourf(xx, yy, classification_plane, cmap=cmap)
plt.plot(x1[:, 0], x1[:, 1], 'o', c="red")
plt.plot(x2[:, 0], x2[:, 1], 'o', c="blue")
plt.show()
