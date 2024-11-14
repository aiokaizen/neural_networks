import json
import sys
import os
import random
import pygame

from neural_network import NeuralNetwork
from training import training_data
from loader import load


def main():
    screen_w, screen_h = 400, 400
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Neural networks")
    clock = pygame.time.Clock()
    pygame.init()
    nn_filename = 'nn3.json'

    if os.path.exists(nn_filename):
        nn = NeuralNetwork.load(nn_filename)
    else:
        nn = NeuralNetwork(784, [65, 28], 10)

    running = True

    # print(f"Loading data...")
    # images, labels = load('training')

    # print('Processing training data...')
    # for i, pixels in enumerate(images):
    #     images[i] = [pxl / 255 for pxl in pixels]  # Reduce the color value to a range of (0, 1)
    #     label = labels[i]
    #     labels[i] = [1 if label == i else 0 for i in range(10)]

    # test_images, test_labels = load('test')
    # for i, pixels in enumerate(test_images):
    #     test_images[i] = [pxl / 255 for pxl in pixels]  # Reduce the color value to a range of (0, 1)

    # # Train the network
    # open('_stop', 'w').close()
    # for _ in range(100):
    #     # Load settings
    #     settings = {}
    #     with open('settings.json', 'r') as f:
    #         settings = json.load(f)


    #     print('Start training...')
    #     lr = settings.get('learning_rate')
    #     if lr:
    #         nn.learning_rate = lr
    #     nn.train(
    #         images[:settings.get('training_pool_size', 60000)],
    #         labels[:settings.get('training_pool_size', 60000)],
    #         settings.get('batch_size', 10),
    #         nn_filename
    #     )
    #     nn.test(
    #         test_images[:settings.get('testing_pool_size', 10000)],
    #         test_labels[:settings.get('testing_pool_size', 10000)],
    #         get_prediction,
    #         {
    #             "batch_size": settings.get('batch_size', 10),
    #             "pool_size": settings.get("training_pool_size")
    #         }
    #     )

    #     # Randomizing data after each epoche
    #     training_zipped = list(zip(images, labels))
    #     random.shuffle(training_zipped)
    #     images, labels = zip(*training_zipped)
    #     images = list(images)
    #     labels = list(labels)

    #     # Stop the program by creating a file _stop file to stop
    #     if os.path.exists('./stop'):
    #         os.remove('./stop')
    #         break

    #     beep = pygame.mixer.Sound('beep.wav')
    #     beep.set_volume(0.05)
    #     beep.play()

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill('#eeeeee')

        # for _ in range(100):
        #     random.shuffle(training_data)
        #     for data in training_data:
        #         nn.train(data['inputs'], data['targets'])

        res = 10  # Resolution
        rows = screen_w // res
        cols = screen_h // res
        for i in range(cols):
            for j in range(rows):
                x1 = i / cols
                x2 = j / rows
                print(x1, '|', x2)
                y = nn.predict([x1, x2])[0]
                grey_scale = y * 255
                red, green, blue = grey_scale, grey_scale, grey_scale
                pygame.draw.rect(screen, [red, green, blue], [i * res, j * res, res, res])

        pygame.display.flip()
        clock.tick(10)


def get_prediction(outputs):
    max_value = 0
    prediction = -1
    for i, output in enumerate(outputs):
        if output > max_value:
            max_value = output
            prediction = i
    return prediction


if __name__ == "__main__":
    main()
