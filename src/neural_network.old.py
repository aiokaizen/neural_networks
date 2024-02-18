import math
import numpy as np

from matrix import Matrix


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def dsigmoid(y):
    # Returns the derivitive of the sigmoid function
    # S'(x) = S(x) * (1 - S(x))
    # Given that y is the result of the sigmoid function
    # y = sigmoid(x)
    return y * (1 - y)


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = 0.1

        # Input > Hidden
        self.weights_ih = Matrix(self.hidden_nodes, self.input_nodes)
        # hidden > Output
        self.weights_ho = Matrix(self.output_nodes, self.hidden_nodes)

        self.bias_h = Matrix(self.hidden_nodes, 1, 1)
        self.bias_o = Matrix(self.output_nodes, 1, 1)

        # Randomize the weights and biases
        self.weights_ih.randomize()
        self.weights_ho.randomize()
        self.bias_h.randomize()
        self.bias_o.randomize()
    
    def predict(self, input_array):

        # Feed forward

        input = Matrix.from_array(input_array)

        # Calculate the matrix product of the input and the hidden weights
        self.hidden_layer = Matrix.matrix_product(self.weights_ih, input)
        # Add the hidden biases
        self.hidden_layer.add(self.bias_h)
        # Apply the activation function
        self.hidden_layer.map(sigmoid)

        # Pass the data to the output layer
        # This is done by calculating the matrix (dot) product of the hidden layer with
        # ho weights.
        self.output_layer = Matrix.matrix_product(self.weights_ho, self.hidden_layer)
        self.output_layer.add(self.bias_o)
        self.output_layer.map(sigmoid)
        return self.output_layer.flatten()
    
    def train(self, inputs, targets):
        
        self.predict(inputs)

        inputs = Matrix.from_array(inputs)
        outputs = self.output_layer
        targets = Matrix.from_array(targets)

        output_errors = Matrix.subtract(targets, outputs)
        # Calculate gradient
        # G = learning_rate * output_error * (outputs * (1 - outputs)) * hidden_layer_transposed
        gradients_ho = outputs.copy()
        gradients_ho.map(dsigmoid)
        gradients_ho.multiply(output_errors)
        gradients_ho.multiply(self.learning_rate)
        inputs_t = Matrix.transpose(self.hidden_layer)
        delta_ho = Matrix.matrix_product(gradients_ho, inputs_t)
        self.weights_ho.add(delta_ho)
        self.bias_o.add(gradients_ho)

        # Calculate the hidden errors
        who_t = Matrix.transpose(self.weights_ho)  # Trnasposed hidden to output weights
        # Calculated from back propagation
        hidden_errors = Matrix.matrix_product(who_t, output_errors)
        # Calculate gradient
        # G = learning_rate * output_error * (outputs * (1 - outputs)) * hidden_layer_transposed
        gradients_ih = self.hidden_layer.copy()
        gradients_ih.map(dsigmoid)
        gradients_ih.multiply(hidden_errors)
        gradients_ih.multiply(self.learning_rate)
        inputs_t = Matrix.transpose(inputs)
        delta_ih = Matrix.matrix_product(gradients_ih, inputs_t)
        self.weights_ih.add(delta_ih)
        self.bias_h.add(gradients_ih)