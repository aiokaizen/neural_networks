import sys
import pygame

from settings import *
from perceptron import Perceptron
from training import Point

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Neural networks")
    clock = pygame.time.Clock()
    pygame.init()
    running = True

    # Create training dataset
    points = []
    for _ in range(100):
        points.append(Point())
    
    perceptron = Perceptron([0, 0, 1])
    training_index = 0

    draw_results = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # if event.type == pygame.MOUSEBUTTONUP:
            #     train(perceptron, points)

        point = points[training_index]
        inputs = [point.x, point.y, point.bias]
        answer = point.label
        perceptron.train(inputs, answer)
        training_index += 1
        if training_index == len(points):
            training_index = 0
            print('weights:', perceptron.weights)
        
        if draw_results:

            screen.fill('#eeeeee')

            draw(perceptron, points)

            pygame.display.flip()

            clock.tick(10)


def train(perceptron, points):
    for point in points:
        inputs = [point.x, point.y, point.bias]
        answer = point.label
        perceptron.train(inputs, answer)
    print('weights:', perceptron.weights)


def draw(perceptron, points):
    draw_bg(perceptron)

    guesses = []
    for point in points:
        inputs = [point.x, point.y, point.bias]
        guesses.append(perceptron.guess(inputs))

    for index, point in enumerate(points):
        guess = guesses[index]
        point.draw(guess)


def draw_bg(perceptron):
    screen = pygame.display.get_surface()
    # x_axis_start = SCREEN_WIDTH / 2, 0
    # x_axis_end = SCREEN_WIDTH / 2, SCREEN_HEIGHT
    # y_axis_start = 0, SCREEN_HEIGHT / 2
    # y_axis_end = SCREEN_WIDTH, SCREEN_HEIGHT / 2
    # pygame.draw.line(screen, '#555555', x_axis_start, x_axis_end, 1)
    # pygame.draw.line(screen, '#555555', y_axis_start, y_axis_end, 1)
    
    # draw separation line
    p1 = Point(-1, f(-1))
    p2 = Point(1, f(1))
    pygame.draw.line(screen, '#555555', (p1.px(), p1.py()), (p2.px(), p2.py()), 1)
    
    # draw perceptron guess line
    p1 = Point(-1, perceptron.guess_y(-1))
    p2 = Point(1, perceptron.guess_y(1))
    pygame.draw.line(screen, 'red', (p1.px(), p1.py()), (p2.px(), p2.py()), 1)


if __name__ == "__main__":
    main()
