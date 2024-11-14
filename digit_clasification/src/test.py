import json
import os
from neural_network import NeuralNetwork
from loader import load


def test_nn(filename='nn.json', pool_size=1_000):

    print('Initiate testing for neural network loaded from:', filename)
    nn = NeuralNetwork.load(filename)

    score = 0
    images, labels = load('test')
    pool_size = min(len(labels), pool_size)
    for i, pixels in enumerate(images[:pool_size]):
        pixels = [p / 255 for p in pixels]  # Reduce the color value to a range of (0, 1)
        outputs = nn.predict(pixels)
        prediction = get_prediction(outputs)
        if prediction == labels[i]:
            score += 1
    score = score * 100 / pool_size
    print('Test complete.\n')
    print('Score:', str(score) + '%')
    data = []
    if os.path.exists('test.json'):
        with open('test.json', 'r') as f:
            data = json.load(f)
    with open('test.json', 'a') as f:
        data.append({
            'neural_network': {
                'hidden_nodes': nn.hidden_nodes,
                'learning_rate': nn.learning_rate
            },
            'score': f'{score}%',
        })
        json.dump(data, f, indent=4)
    
    return score




def get_prediction(outputs):
    max_value = 0
    prediction = -1
    for i, output in enumerate(outputs):
        if output > max_value:
            max_value = output
            prediction = i
    return prediction


if __name__ == "__main__":
    test_nn('nn3.json')
