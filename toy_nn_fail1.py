import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(1)


class SigmoidActivation:
    def __call__(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def deriv(self, x):
        return x * (1.0 - x)


class IdentityActivation:
    def __call__(self, x):
        return x

    def deriv(self, x):
        return 1.0


class SquareLoss:
    def __call__(self, true, pred):
        return np.sum((pred - true) ** 2)

    def deriv(self, true, pred):
        return 2 * np.sum(pred - true)


square_loss = SquareLoss()


class CrossEntropyLoss:
    def __call__(self, true, pred):
        return -np.sum( true * np.log(pred) + (1 - true) * np.log(1 - pred))

    def deriv(self, true, pred):
        return (pred - true) / (pred * (1 - pred))


ce_loss = CrossEntropyLoss()


class BaseNeuron:
    def __init__(self, n_input, activation_fun=IdentityActivation()):
        self.activation_fun = activation_fun
        # self.bias = random.random()
        self.bias = 0.1

        self.weights = np.ndarray(n_input)
        for i in range(n_input):
            # self.weights[i] = random.random()
            self.weights[i] = 0.1
        self.output = None
        self.weights_error_grad = 0
        self.bias_error_grad = 0

    def activate(self, inputs):
        act = np.sum(self.weights * inputs) + self.bias
        self.output = self.activation_fun(act)
        return self

    def update_error_grad(self, inputs, error, reset):
        if reset:
            self.bias_error_grad = 0
            self.weights_error_grad = 0
        self.bias_error_grad += error * self.activation_fun.deriv(self.output)
        self.weights_error_grad += self.bias_error_grad * inputs
        # print("eg: %s %s " % (self.bias_error_grad, self.weights_error_grad))

    def update_weights(self, learning_rate):
        print(self.weights_error_grad)
        self.weights -= learning_rate * self.weights_error_grad
        self.bias -= learning_rate * self.bias_error_grad
        return self

    def __str__(self):
        return ("bias: %s | weights: %s | " % (self.bias, self.weights) +
                "output: %s | dw: %s | " % (self.output, self.weights_error_grad) +
                "db: %s" % self.bias_error_grad)

    def __repr__(self):
        return self.__str__()


class SimpleNn:
    def __init__(self, n_input, n_output, loss):
        self.n_input = n_input
        self.n_output = n_output
        self.loss = loss
        self.hidden_layers = []
        self.output_layer = None

    def add_hidden_layer(self, n_neurons, neuron_class):
        if len(self.hidden_layers) == 0:
            layer = [neuron_class(self.n_input) for i in range(n_neurons)]
        else:
            n_output_last_hidden_layer = len(self.hidden_layers[-1])
            layer = [neuron_class(n_output_last_hidden_layer) for i in range(n_neurons)]
        self.hidden_layers.append(layer)

    def add_output_layer(self, n_neurons, neuron_class):
        if len(self.hidden_layers) == 0:
            layer = [neuron_class(self.n_input) for i in range(n_neurons)]
        else:
            n_output_last_hidden_layer = len(self.hidden_layers[-1])
            layer = [neuron_class(n_output_last_hidden_layer) for i in range(n_neurons)]
        self.output_layer = layer
        self.n_output = n_neurons

    def forward_prop(self, inputs):
        output = np.ndarray((len(inputs), self.n_output))
        for i, sample in enumerate(inputs):
            h_sample = sample
            for layer in self.hidden_layers:
                fed_sample = []
                for neuron in layer:
                    fed_sample.append(neuron.activate(h_sample).output)
                h_sample = np.array(fed_sample)

            sample_output = np.ndarray(self.n_output)
            for j, neuron in enumerate(self.output_layer):
                neuron.activate(sample)
                sample_output[j, ] = neuron.output
            output[i, ] = sample_output
        return output

    def backprop(self, inputs, output, true_output):
        error = self.loss(true_output, output)
        d_error = self.loss.deriv(true_output, output)
        print(error, d_error)

        for i, sample in enumerate(inputs):
            for neuron in self.output_layer:
                neuron.update_error_grad(sample, d_error, reset = i == 0)
            prev_layer = self.output_layer

            for layer in reversed(self.hidden_layers):
                for j, neuron in enumerate(layer):
                    h_error = 0.0
                    for prev_neuron in prev_layer:
                        h_error += prev_neuron.weights[j] * prev_neuron.error_grad
                    neuron.update_error_grad(sample, h_error, reset = i == 0)
                prev_layer = layer
        return error

    def update_weights(self, learning_rate):
        for layer in self.hidden_layers:
            for neuron in layer:
                neuron.update_weights(learning_rate)
        for neuron in self.output_layer:
            neuron.update_weights(learning_rate)

    def __str__(self):
        res = []
        for i in range(len(self.hidden_layers)):
            res.append("Layer %s" % i)
            for neuron in self.hidden_layers[i]:
                res.append("    %s" % str(neuron))
        res.append("Output layer")
        for neuron in self.output_layer:
            res.append("    %s" % str(neuron))
        return "\n".join(res)


snn = SimpleNn(1, 1, square_loss)
snn.add_output_layer(1, BaseNeuron)

# snn.hidden_layers[0][0].activate(np.array([1]))
print(snn)
print("--")

X = np.array([np.array(random.random()) for i in range(20)])
Y = np.array([np.array(x + random.normalvariate(0, 0.1)) for x in X]).reshape(-1, 1)
Y /= np.max(Y)
# plt.scatter(X, Y)
# plt.show()

errors = []
print(snn)
for epoch in range(2):
    output = snn.forward_prop(X)
    error = snn.backprop(X, output, Y)
    errors.append(error)
    snn.update_weights(0.05)
    print(snn)


# plt.plot(errors)
# plt.show()
