import math
import json

from matrix import Matrix


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def dsigmoid(y):
    # Returns the derivitive of the sigmoid function
    # S'(x) = S(x) * (1 - S(x))
    # Given that y is the result of the sigmoid function
    # y = sigmoid(x)
    return y * (1 - y)


class Layer:

    def __init__(self, nodes, prev_layer_nodes):
        self.nodes = Matrix(nodes, 1)
        self.weights = Matrix(nodes, prev_layer_nodes)
        self.biases = Matrix(nodes, 1, default=1)
        self.errors = Matrix(nodes, 1)
        self.gradients = Matrix(nodes, 1)
    
    def randomize(self):
        self.weights.randomize()
        self.biases.randomize()
    
    def count(self):
        return len(self.nodes.data)


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        """ hidden_nodes: [
           3 (3 nodes for the first hidden layer),
           4 (4 nodes for the second hidden layer),
           ... 
        ]"""
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = 0.1

        self.hidden_layers = []
        for index, nodes in enumerate(self.hidden_nodes):
            if index > 0:
                prev_layer_nodes = self.hidden_nodes[index - 1]
            else:
                prev_layer_nodes = self.input_nodes
            self.hidden_layers.append(Layer(nodes, prev_layer_nodes))

        # Last hidden > Output
        self.output_layer = Layer(self.output_nodes, self.hidden_nodes[-1])

        # Randomize the weights and biases of all the layers
        for layer in self.hidden_layers:
            layer.randomize()
        self.output_layer.randomize()
    
    def predict(self, input_array):

        # Feed forward

        inputs = Matrix.from_array(input_array)

        # Calculate the matrix product of the input and the hidden weights
        for i, hlayer in enumerate(self.hidden_layers):
            prev_layer = self.hidden_layers[i - 1].nodes if i > 0 else inputs
            hlayer.nodes = Matrix.matrix_product(hlayer.weights, prev_layer)
            hlayer.nodes.add(hlayer.biases)
            # Apply the activation function
            hlayer.nodes.map(sigmoid)

        # Pass the data to the output layer
        # This is done by calculating the matrix (dot)
        # product of the last hidden layer with ho weights.
        self.output_layer.nodes = Matrix.matrix_product(
            self.output_layer.weights, self.hidden_layers[-1].nodes
        )
        self.output_layer.nodes.add(self.output_layer.biases)
        self.output_layer.nodes.map(sigmoid)

        return self.output_layer.nodes.flatten()
    
    def train(self, inputs, targets):
        
        self.predict(inputs)

        inputs = Matrix.from_array(inputs)
        outputs = self.output_layer.nodes
        targets = Matrix.from_array(targets)

        output_errors = Matrix.subtract(targets, outputs)

        # Calculate gradient
        # G = learning_rate * output_error * (outputs * (1 - outputs)) * hidden_layer_transposed
        gradients_ho = outputs.copy()
        gradients_ho.map(dsigmoid)
        gradients_ho.multiply(output_errors)
        gradients_ho.multiply(self.learning_rate)
        inputs_t = Matrix.transpose(self.hidden_layers[-1].nodes)
        delta_ho = Matrix.matrix_product(gradients_ho, inputs_t)
        self.output_layer.weights.add(delta_ho)
        self.output_layer.biases.add(gradients_ho)
        self.output_layer.errors = output_errors
        self.output_layer.gradients = gradients_ho

        # Calculate the hidden errors
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            layer = self.hidden_layers[i]
            prev_layer = self.hidden_layers[i - 1].nodes if i > 0 else inputs
            next_layer = self.hidden_layers[i + 1] if i < len(self.hidden_layers) - 1 else self.output_layer
            nlw_t = Matrix.transpose(next_layer.weights)  # Trnasposed next layer's weights
            # Calculated by back propagation
            layer.errors = Matrix.matrix_product(nlw_t, next_layer.errors)
            # Calculate gradient
            # G = learning_rate * output_error * (outputs * (1 - outputs)) * hidden_layer_transposed
            layer.gradients = layer.nodes.copy()
            layer.gradients.map(dsigmoid)
            layer.gradients.multiply(layer.errors)
            layer.gradients.multiply(self.learning_rate)
            inputs_t = Matrix.transpose(prev_layer)
            delta_ih = Matrix.matrix_product(layer.gradients, inputs_t)
            layer.weights.add(delta_ih)
            layer.biases.add(layer.gradients)


    def save(self):
        with open('nn.json', 'w') as f:
            data = {
                'input_nodes': self.input_nodes,
                'hidden_nodes': self.hidden_nodes,
                'output_nodes': self.output_nodes,
                'learning_rate': self.learning_rate,
                'hidden_layers': [
                    {
                        'weights': layer.weights.data,
                        'biases': layer.biases.data,
                    } for layer in self.hidden_layers
                ],
                'output_layer': {
                    'weights': self.output_layer.weights.data,
                    'biases': self.output_layer.biases.data,
                }
            }
            f.write(json.dumps(data, indent=4))
    
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
            nn = NeuralNetwork(data['input_nodes'], data['hidden_nodes'], data['output_nodes'])
            nn.learning_rate = data['learning_rate']
            for i, layer in enumerate(data['hidden_layers']):
                nn.hidden_layers[i].weights.data = layer['weights']
                nn.hidden_layers[i].biases.data = layer['biases']
            nn.output_layer.weights.data = data['output_layer']['weights']
            nn.output_layer.biases.data = data['output_layer']['biases']
            return nn
