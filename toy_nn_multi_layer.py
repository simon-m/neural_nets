
import sys
import itertools
import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, colorConverter
from mpl_toolkits.mplot3d import Axes3D

# Todo: add tanh and softplus

def logistic(z):
    return 1 / (1 + np.exp(-z))

# derivative with respect to Zh, expressed as  H = softmax(Zh)
def d_logistic(H):
    return H * (1 - H)

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

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)

# derivative with respect to z, expressed as  y = softmax(z)
def d_softmax(Y):
    return Y * (1 - Y)

def cross_entropy_loss(Y, T):
    return - np.sum(T * np.log(Y))

# derivative with respect to output weights
def d_cross_entropy_loss(Y, T):
    return (Y - T) / (Y * (1 - Y))


class NNLayer(object):
    def get_activations(self, X):
        raise NotImplemented

    def get_param_grads(self, X, output_grads):
        return []

    def get_input_grads(self, Y, output_grads):
        raise NotImplemented

    def get_params(self):
        return []

    def update_params(self, param_grads, learning_rate):
        pass


class NNLinearLayer(NNLayer):
    def __init__(self, n_input, n_output):
        self.W = np.random.randn(n_input, n_output) * 0.1
        self.b = np.zeros(n_output)

    def get_activations(self, X):
        return X.dot(self.W) + self.b

    def get_param_grads(self, X, output_grads):
        JW = X.T.dot(output_grads)
        Jb = np.sum(output_grads, axis=0)
        # return itertools.chain(np.nditer(JW), np.nditer(Jb))
        return JW, Jb

    def get_input_grads(self, Y, output_grads):
        return output_grads.dot(self.W.T)

    def update_params(self, param_grads, learning_rate):
        JW, Jb = param_grads
        # print(self.W)
        self.W -= learning_rate * JW
        # print("\n")
        # print(self.W)
        # print("--")
        self.b -= learning_rate * Jb

    def __str__(self):
        return "\n".join([str(self.W), str(self.b)])


class NNLogisticLayer(NNLayer):
    def get_activations(self, X):
        return logistic(X)

    def get_input_grads(self, Y, output_grads):
        return output_grads * d_logistic(Y)


class NNReluLayer(NNLayer):
    def get_activations(self, X):
        return relu(X)

    def get_input_grads(self, Y, output_grads):
        return output_grads * d_relu(Y)


class NNSoftmaxOutputLayer(NNLayer):
    def get_activations(self, X):
        return softmax(X)

    def get_input_grads(self, Y, T):
        return (Y - T) / Y.shape[0]

    def get_loss(self, Y, T):
        return cross_entropy_loss(Y, T) / Y.shape[0]


class ToyNN(object):
    def __init__(self, batch_size = 25,
                 n_epochs=10,
                 learning_rate=0.01,
                 learning_rate_update_fun=lambda x: x):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = learning_rate
        self.lr_update = learning_rate_update_fun
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        in_act = X
        activations = [X]
        for layer in self.layers:
            out_act = layer.get_activations(in_act)
            activations.append(out_act)
            in_act = out_act
        return activations

    def backprop(self, activations, T):
        out_grad = T
        param_grads = []
        for layer in reversed(self.layers):
            out_act = activations.pop()
            in_grad = layer.get_input_grads(out_act, out_grad)

            in_act = activations[-1]
            p_grads = layer.get_param_grads(in_act, out_grad)
            param_grads.append(p_grads)

            out_grad = in_grad
        return list(reversed(param_grads))

    def update_layer_params(self, param_grads, learning_rate):
        for layer, layer_param_grads in zip(self.layers, param_grads):
            layer.update_params(layer_param_grads, learning_rate)

    def train_one_epoch(self, batches):
        for X, T in batches:
            activations = self.forward(X)
            param_grads = self.backprop(activations, T)
            self.update_layer_params(param_grads, self.lr)
            self.lr = self.lr_update(self.lr)

    def predict_proba(self, X):
        return self.forward(X)[-1]

    def predict(self, X):
        return np.around(self.predict_proba(X))

    def train(self, X, T, Xv, Tv):
        n_batches = X.shape[0] // self.batch_size
        batches = list(zip(np.array_split(X, n_batches, axis=0),
                      np.array_split(T, n_batches, axis=0)))

        train_losses = []
        valid_losses = []
        n_iter = 1
        for i in range(self.n_epochs):
            self.train_one_epoch(batches)
            train_preds = self.predict_proba(X)
            train_losses.append(self.layers[-1].get_loss(train_preds, T))

            valid_preds = self.predict_proba(Xv)
            valid_losses.append(self.layers[-1].get_loss(valid_preds, Tv))

            if i > 2 and (valid_losses[-1] >= valid_losses[-2] >= valid_losses[-3]):
               break

            n_iter += 1

        return train_losses, valid_losses, n_iter


# load the data from scikit-learn.
digits = datasets.load_digits()

# Load the targets.
# Note that the targets are stored as digits, these need to be
#  converted to one-hot-encoding for the output sofmax layer.
T = np.zeros((digits.target.shape[0], 10))
T[np.arange(len(T)), digits.target] += 1

# Divide the data into a train and test set.
X_train, X_test, T_train, T_test = model_selection.train_test_split(
    digits.data, T, test_size=0.5)
# Divide the test set into a validation set and final test set.
X_validation, X_test, T_validation, T_test = model_selection.train_test_split(
    X_test, T_test, test_size=0.5)


batch_size = 25
n_epochs = 100
lr = 0.1
lr_update_factor = lr / (n_epochs * batch_size)
tnn_logit = ToyNN(batch_size, n_epochs, lr, lambda x: x - lr_update_factor)

tnn_logit.add_layer(NNLinearLayer(X_train.shape[1], 20))
tnn_logit.add_layer(NNLogisticLayer())
tnn_logit.add_layer(NNLinearLayer(20, 20))
tnn_logit.add_layer(NNLogisticLayer())
tnn_logit.add_layer(NNLinearLayer(20, T_train.shape[1]))
tnn_logit.add_layer(NNSoftmaxOutputLayer())

tloss, vloss, i = tnn_logit.train(X_train, T_train, X_validation, T_validation)

plt.figure(1)
plt.plot(range(i), tloss, "b-")
plt.plot(range(i), vloss, "r-")

# Get results of test data
y_true = np.argmax(T_test, axis=1)  # Get the target outputs
y_pred = np.argmax(tnn_logit.predict_proba(X_test), axis=1)  # Get the predictions made by the network
test_accuracy = metrics.accuracy_score(y_true, y_pred)  # Test set accuracy
print('The accuracy on the test set is {:.2f}'.format(test_accuracy))

##

batch_size = 50
n_epochs = 100
lr = 0.05
lr_update_factor = lr / (n_epochs * batch_size)
tnn_relu = ToyNN(batch_size, n_epochs, lr, lambda x: x - lr_update_factor)

tnn_relu.add_layer(NNLinearLayer(X_train.shape[1], 25))
tnn_relu.add_layer(NNReluLayer())
tnn_relu.add_layer(NNLinearLayer(25, 50))
tnn_relu.add_layer(NNReluLayer())
tnn_relu.add_layer(NNLinearLayer(50, 20))
tnn_relu.add_layer(NNReluLayer())
tnn_relu.add_layer(NNLinearLayer(20, T_train.shape[1]))
tnn_relu.add_layer(NNSoftmaxOutputLayer())

tloss, vloss, i = tnn_relu.train(X_train, T_train, X_validation, T_validation)

plt.figure(2)
plt.plot(range(i), tloss, "b-")
plt.plot(range(i), vloss, "r-")

# Get results of test data
y_true = np.argmax(T_test, axis=1)  # Get the target outputs
y_pred = np.argmax(tnn_relu.predict_proba(X_test), axis=1)  # Get the predictions made by the network
test_accuracy = metrics.accuracy_score(y_true, y_pred)  # Test set accuracy
print('The accuracy on the test set is {:.2f}'.format(test_accuracy))

plt.show()


