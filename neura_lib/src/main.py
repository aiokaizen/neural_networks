import sys
import random
import pygame

from neural_network import NeuralNetwork
from training import training_data


def main():
    screen_w, screen_h = 400, 400
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Neural networks")
    clock = pygame.time.Clock()
    pygame.init()

    nn = NeuralNetwork(2, [6, 4, 2], 1)
    # nn = NeuralNetwork.load('nn.json')

    running = True

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill('#eeeeee')
        
        for _ in range(500):
            random.shuffle(training_data)
            for data in training_data:
                nn.train(data['inputs'], data['targets'])

        res = 5  # Resolution
        rows = screen_w // res
        cols = screen_h // res
        for i in range(cols):
            for j in range(rows):
                x1 = i / cols
                x2 = j / rows
                y = nn.predict([x1, x2])[0]
                grey_scale = y * 255
                red, green, blue = grey_scale, grey_scale, grey_scale
                pygame.draw.rect(screen, [red, green, blue], [i * res, j * res, res, res])
        
        # if nn.predict([0, 0])[0] < 0.04 and nn.predict([0, 1])[0] > 0.95:
        #     nn.save()
        #     pygame.quit()
        #     sys.exit()

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
