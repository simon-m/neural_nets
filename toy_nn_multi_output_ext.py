# http://peterroelants.github.io/posts/neural_network_implementation_part04/

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, colorConverter
from mpl_toolkits.mplot3d import Axes3D


def logistic(z):
    return 1 / (1 + np.exp(-z))

# derivative with respect to Zh, expressed as  H = softmax(Zh)
def d_logistic(Z):
    return logistic(Z) * (1 - logistic(Z))
    # return H * (1 - H)


def gaussian_rbf(x):
    return np.exp(- x ** 2)

def d_gaussian_rbf(x):
    return - 2 * x * gaussian_rbf(x)

def relu(z):
    return np.maximum(np.zeros(z.shape), z)

def d_relu(z):
    zn = z.copy()
    zn[zn > 0] = 1
    zn[zn < 0] = 0
    return zn

hidden_activation = relu
d_hidden_activation = d_relu

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)

# derivative with respect to z, expressed as  y = softmax(z)
def d_softmax(Z):
    return softmax(Z) * (1 - softmax(Z))
    # return Y * (1 - Y)

output_activation = softmax
d_output_activation = d_softmax

def cross_entropy_loss(Y, T):
    return - np.sum(T * np.log(Y))

# derivative with respect to output weights
def d_cross_entropy_loss(Y, T):
    return (Y - T) / (Y * (1 - Y))

loss = cross_entropy_loss
d_loss = d_cross_entropy_loss

def linear_activate_hidden(X, Wh, bh):
    return X.dot(Wh) + bh

def activate_hidden(Hl):
    return hidden_activation(Hl)

def linear_activate_output(H, Wo, bo):
    return H.dot(Wo) + bo

def activate_output(Ho):
    return output_activation(Ho)

# Interface
def mlmo_nn_predict_proba(X, Wh, bh, Wo, bo):
    return activate_output(
            linear_activate_output(
                activate_hidden(
                    linear_activate_hidden(X, Wh, bh)), Wo, bo))

def mlmo_nn_predict(X, Wh, bh, Wo, bo):
    return np.around(mlmo_nn_predict_proba(X, Wh, bh, Wo, bo))

# Errors
def output_error(Y, T, Yl):
    return d_loss(Y, T) * d_output_activation(Yl)

def hidden_error(Hl, Wo, Eo):
    return d_hidden_activation(Hl) * (Eo.dot(Wo.T))

# Error gradient wrt weight/bias for each sample
def output_weights_grad(H, Eo):
    return H.T.dot(Eo)

def output_bias_grad(Eo):
    return np.sum(Eo, axis=0, keepdims=True)

def hidden_weights_grad(X, Eh):
    return X.T.dot(Eh)

def hidden_bias_grad(Eh):
    return np.sum(Eh, axis=0, keepdims=True)

# Update for the weight according to the error gradient
def output_weights_update(H, Eo, learning_rate):
    return learning_rate * output_weights_grad(H, Eo)

def hidden_weights_update(X, Eh, learning_rate):
    return learning_rate * hidden_weights_grad(X, Eh)

# Update for the bias according to the error gradient
def output_bias_update(Eo, learning_rate):
    return learning_rate * output_bias_grad(Eo)

def hidden_bias_update(Eh, learning_rate):
    return learning_rate * hidden_bias_grad(Eh)


def backprop_gradients(X, T, Wh, bh, Wo, bo):
    Hl = linear_activate_hidden(X, Wh, bh)
    H = activate_hidden(Hl)
    Yl = linear_activate_output(H, Wo, bo)
    Y = activate_output(Yl)

    Eo = output_error(Y, T, Yl)
    Eh = hidden_error(Hl, Wo, Eo)

    JWo = output_weights_grad(H, Eo)
    Jbo = output_bias_grad(Eo)
    JWh = hidden_weights_grad(X, Eh)
    Jbh = hidden_bias_grad(Eh)

    return [JWh, Jbh, JWo, Jbo]

def update_velocity(gradients, velocities, momentum, learning_rate):
    return [momentum * v - learning_rate * g for v, g in zip(velocities, gradients)]

def update_params(params, velocities):
    return [p + v for p,v in zip(params, velocities)]


X, t = datasets.make_circles(n_samples=100, shuffle=False, factor=0.3, noise=0.1)
# Separate the red and blue points for plotting
x_red = X[t == 0]
x_blue = X[t == 1]

T = np.zeros((100, 2)) # Define target matrix
T[t == 1, 1] = 1
T[t == 0, 0] = 1

# Initialize weights and biases
init_var = 1
# Initialize hidden layer parameters
bh = np.random.randn(1, 3) * init_var
Wh = np.random.randn(2, 3) * init_var
# Initialize output layer parameters
bo = np.random.randn(1, 2) * init_var
Wo = np.random.randn(3, 2) * init_var

Hl = linear_activate_hidden(X, Wh, bh)
H = activate_hidden(Hl)
Yl = linear_activate_output(H, Wo, bo)
Y = activate_output(Yl)

Eo = output_error(Y, T, Yl)
Eh = hidden_error(Hl, Wo, Eo)

# Compute the gradients by backpropagation
# Compute the gradients of the output layer
JWo = output_weights_grad(H, Eo)
Jbo = output_bias_grad(Eo)
# Compute the gradients of the hidden layer
JWh = hidden_weights_grad(X, Eh)
Jbh = hidden_bias_grad(Eh)


# Combine all parameter matrices in a list
params = [Wh, bh, Wo, bo]
# Combine all parameter gradients in a list
grad_params = [JWh, Jbh, JWo, Jbo]

# Set the small change to compute the numerical gradient
eps = 0.0001

# Check each parameter matrix
for p_idx in range(len(params)):
    # Check each parameter in each parameter matrix
    for row in range(params[p_idx].shape[0]):
        for col in range(params[p_idx].shape[1]):
            # Copy the parameter matrix and change the current parameter slightly
            p_matrix_min = params[p_idx].copy()
            p_matrix_min[row, col] -= eps
            p_matrix_plus = params[p_idx].copy()
            p_matrix_plus[row, col] += eps
            # Copy the parameter list, and change the updated parameter matrix
            params_min = params[:]
            params_min[p_idx] = p_matrix_min
            params_plus = params[:]
            params_plus[p_idx] = p_matrix_plus
            # Compute the numerical gradient
            grad_num = (cross_entropy_loss(mlmo_nn_predict_proba(X, *params_plus), T) -
                        cross_entropy_loss(mlmo_nn_predict_proba(X, *params_min), T)) / \
                       (2 * eps)
            # Raise error if the numerical grade is not close to the backprop gradient
            if not np.isclose(grad_num, grad_params[p_idx][row, col]):
                msg = 'Numerical gradient of {:.6f} is not close to the backpropagation gradient of {:.6f}!'.format(float(grad_num), float(grad_params[p_idx][row,col]))
                raise ValueError(msg)
                # print(msg)

print('No gradient errors found')


param_velocities = [np.zeros(p.shape) for p in params]
momentum = 0.6
learning_rate = 0.005
n_epochs = 1000

lr_update = learning_rate / n_epochs
losses = [cross_entropy_loss(mlmo_nn_predict_proba(X, *params), T)]

for i in range(n_epochs):
    param_gradients = backprop_gradients(X, T, *params)
    param_velocities = update_velocity(param_gradients, param_velocities, momentum, learning_rate)
    params = update_params(params, param_velocities)
    losses.append(cross_entropy_loss(mlmo_nn_predict_proba(X, *params), T))

plt.figure(1)
plt.plot(losses)

# Plot the resulting decision boundary
# Generate a grid over the input space to plot the color of the
#  classification at that grid point
nb_of_xs = 200
xs1 = np.linspace(-2, 2, num=nb_of_xs)
xs2 = np.linspace(-2, 2, num=nb_of_xs)
xx, yy = np.meshgrid(xs1, xs2) # create the grid
# Initialize and fill the classification plane
classification_plane = np.zeros((nb_of_xs, nb_of_xs))

for i in range(nb_of_xs):
    for j in range(nb_of_xs):
        pred = mlmo_nn_predict(np.array([[xx[i, j], yy[i, j]]]), *params)
        classification_plane[i, j] = pred[0, 0]
# Create a color map to show the classification colors of each grid point
cmap = ListedColormap([
        colorConverter.to_rgba('b', alpha=0.30),
        colorConverter.to_rgba('r', alpha=0.30)])

plt.figure(2)
# Plot the classification plane with decision boundary and input samples
plt.contourf(xx, yy, classification_plane, cmap=cmap)
# Plot both classes on the x1, x2 plane
plt.plot(x_red[:, 0], x_red[:, 1], 'ro', label='class red')
plt.plot(x_blue[:, 0], x_blue[:, 1], 'bo', label='class blue')
plt.grid()
plt.legend(loc=1)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
plt.axis([-1.5, 1.5, -1.5, 1.5])
plt.title('red vs blue classification boundary')
plt.show()
