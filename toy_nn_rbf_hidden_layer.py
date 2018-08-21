# http://peterroelants.github.io/posts/neural_network_implementation_part03/

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Log loss for classification (logistic regression)
def log_loss(preds, truth):
    return - np.sum(truth * np.log(preds) + (1 - truth) * np.log(1 - preds))

def logistic(x):
    return 1 / (1 + np.exp(-x))

def gaussian_rbf(x):
    return np.exp(-x ** 2)

def activate_hidden(x, hidden_weights):
    return gaussian_rbf(x * hidden_weights)

def activate_output(h, output_weights):
    return logistic(h * output_weights)

def rbf_nn_predict_proba(samples, weights_hidden, weights_output):
    return activate_output(activate_hidden(samples, weights_hidden),
                           weights_output)

def rbf_nn_predict(samples, weights_hidden, weights_output):
    return np.around(rbf_nn_predict_proba(samples, weights_hidden, weights_output))

# Error gradient wrt weight for each sample
def output_weights_grad(hidden_activation, preds, truth):
    return (preds - truth).T.dot(hidden_activation)

def hidden_weights_grad(x, weights_hidden, weights_output, preds, truth):
    return np.sum(-2 * x ** 2 * weights_hidden * weights_output *
                  activate_hidden(x, weights_hidden) * (preds - truth))

# Update for the weight according to the error gradient
def output_weights_update(x, preds, truth, learning_rate):
    return learning_rate * output_weights_grad(x, preds, truth)

def hidden_weights_update(x, weights_hidden, weights_output, preds, truth, learning_rate):
    return learning_rate * hidden_weights_grad(x, weights_hidden, weights_output, preds, truth)


# Generate the dataset (1d non-linearly separable classification)
n_samples_per_class = 20
noise_level = 0.5
class_1_mean  = 0
class_2_means = [-2, 2]
x1 = np.random.randn(n_samples_per_class) * noise_level + class_1_mean
x2 = np.random.randn(n_samples_per_class // 2) * noise_level + class_2_means[0]
x3 = np.random.randn(n_samples_per_class // 2) * noise_level + class_2_means[1]

x = np.hstack((x1, x2, x3))
t = np.hstack((np.zeros(n_samples_per_class), np.ones(n_samples_per_class)))

# Plot the dataset
plt.figure(1)
plt.ylim(-1,1)
plt.plot(x1, np.zeros(n_samples_per_class), "r|", ms=30)
plt.plot(x2, np.zeros(n_samples_per_class//2), "b|", ms=30)
plt.plot(x3, np.zeros(n_samples_per_class//2), "b|", ms=30)
# plt.show()

wh, wo = np.random.uniform(-5, 5, 2)
learning_rate = 0.2
n_epochs = 50
lr_update = learning_rate / n_epochs

losses = np.ndarray((n_epochs, 1))
w_cost_iter = [(wh, wo, log_loss(rbf_nn_predict(x, wh, wo), t))]
for i in range(n_epochs):
    preds = rbf_nn_predict_proba(x, wh, wo)
    losses[i] = log_loss(preds, t)
    wo -= output_weights_update(x, preds, t, learning_rate)
    wh -= hidden_weights_update(x, wh, wo, preds, t, learning_rate)
    learning_rate -= lr_update
    # learning_rate /= 1.1
    w_cost_iter.append((wh, wo, log_loss(rbf_nn_predict_proba(x, wh, wo), t)))

plt.figure(2)
plt.plot(losses)
# plt.show()


# Plot the cost in function of the weights
# Define a vector of weights for which we want to plot the cost
nb_of_ws = 200 # compute the cost nb_of_ws times in each dimension
wsh = np.linspace(-10, 10, num=nb_of_ws) # hidden weights
wso = np.linspace(-10, 10, num=nb_of_ws) # output weights
ws_x, ws_y = np.meshgrid(wsh, wso) # generate grid
cost_ws = np.zeros((nb_of_ws, nb_of_ws)) # initialize cost matrix

for i in range(nb_of_ws):
    for j in range(nb_of_ws):
        cost_ws[i, j] = log_loss(rbf_nn_predict_proba(x, ws_x[i, j], ws_y[i, j]) , t)

fig = plt.figure(3)
ax = Axes3D(fig)
surf = ax.plot_surface(ws_x, ws_y, cost_ws, linewidth=0, cmap=cm.pink)
ax.view_init(elev=60, azim=-30)
cbar = fig.colorbar(surf)
cbar.ax.set_ylabel('$\\xi$', fontsize=15)

# Plot the updates
for i in range(1, len(w_cost_iter)):
    wh1, wo1, c1 = w_cost_iter[i-1]
    wh2, wo2, c2 = w_cost_iter[i]
    # Plot the weight-cost value and the line that represents the update
    ax.plot([wh1], [wo1], [c1], 'w+')  # Plot the weight cost value
    ax.plot([wh1, wh2], [wo1, wo2], [c1, c2], 'w-')
# Plot the last weights
wh1, wo1, c1 = w_cost_iter[len(w_cost_iter)-1]
ax.plot([wh1], [wo1], c1, 'w+')

# Show figure
ax.set_xlabel('$w_h$', fontsize=15)
ax.set_ylabel('$w_o$', fontsize=15)
ax.set_zlabel('$\\xi$', fontsize=15)
plt.title('Gradient descent updates on cost surface')
plt.grid()
plt.show()
