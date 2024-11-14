"""
Perceptron is the simplest neural network possible. A computational model of a single neuron.
A perceptron consists of one or more inputs, a processor, and one output.
"""


import random


class Perceptron:

    def __init__(self, inputs, weights=[]):
        self.inputs = inputs
        self.weights = weights
        self.learning_rate = 0.01

        if not self.weights:
            self.init_weights()
        elif len(self.inputs) != len(self.weights):
            raise Exception("The number of inputs should the same as the number of weights.")

    def init_weights(self):
        for _ in range(len(self.inputs)):
            self.weights.append(random.random() * random.choice([-1, 1]))

    def fire(self, inputs=[]):
        sum = 0
        inputs = inputs if inputs else self.inputs
        for input, weight in zip(inputs, self.weights):
            sum += input * weight
        
        return self.activate(sum)
    
    def activate(self, sum):
        return -1 if sum < 0 else 1
    
    def train(self, inputs, answer):
        guess = self.fire(inputs)
        error = answer - guess

        # Tuning the weights
        for index in range(len(self.weights)):
            delta_weight = error * inputs[index] * self.learning_rate
            self.weights[index] += delta_weight
