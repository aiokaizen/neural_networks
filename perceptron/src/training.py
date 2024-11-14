import random
import pygame

from settings import SCREEN_HEIGHT, SCREEN_WIDTH, f, map_range

class Point:

    def __init__(self, x=None, y=None):
        if x is None and y is None:
            self.x = (random.random() - 0.5) * 2  # random int between -1 and 1
            self.y = (random.random() - 0.5) * 2
        elif x is not None and y is not None:
            self.x = x
            self.y = y

        self.bias = 1
        if self.y > f(self.x):
            self.label = 1
        else:
            self.label = -1
    
    def px(self):
        return map_range(self.x, -1, 1, 0, SCREEN_WIDTH)
    
    def py(self):
        return map_range(self.y, -1, 1, SCREEN_HEIGHT, 0)
    
    def draw(self, guess=None):
        position = (self.px(), self.py())
        border = 1 if self.label > 0 else 0
        pygame.draw.circle(pygame.display.get_surface(), "#333333", position, 16, border)

        if guess is not None:
            color = 'lightgreen' if guess == self.label else 'red'
            pygame.draw.circle(pygame.display.get_surface(), color, position, 8)
