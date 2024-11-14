SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 700


def map_range(v, min_1, max_1, min_2, max_2):
   r = (v - min_1) / (max_1 - min_1) * (max_2 - min_2) + min_2
   return r


def f(x):
    # y = ax + b
    return 2 * x - 1.4
