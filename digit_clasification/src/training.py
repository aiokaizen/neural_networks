import random

from neural_network import NeuralNetwork


training_data = [
    {'inputs': [1, 1], 'targets': [0]},
    {'inputs': [1, 0], 'targets': [1]},
    {'inputs': [0, 1], 'targets': [1]},
    {'inputs': [0, 0], 'targets': [0]}
]

def main():
    nn = NeuralNetwork(2, 2, 1)

    for _ in range(10000):
        random.shuffle(training_data)
        for data in training_data:
            nn.train(data['inputs'], data['targets'])

    print(f"result for [0, 0]: {nn.predict([0, 0])}")
    print(f"result for [0, 1]: {nn.predict([0, 1])}")
    print(f"result for [1, 0]: {nn.predict([1, 0])}")
    print(f"result for [1, 1]: {nn.predict([1, 1])}")
    # for data in training_data:
    #     output = nn.feed_forward(data['inputs'])
    #     print(f"result for {data['inputs']}: {output}")


if __name__ == "__main__":
    main()