import math
import numpy as np

from joblib import Parallel, delayed

class InvertedGaussian():
    SIGMA = 64

    def __normalize(self, x: int, y: int, w: int, h: int) -> tuple[float, float]:
        cap = max(w - 1, h - 1)
        x = (2 * (x + (cap - w + 1) / 2) - cap) / cap
        y = (2 * (y + (cap - h + 1) / 2) - cap) / cap

        return x, y
    
    def __compute_pixel(self, x: float, y: float) -> float:
        r = math.sqrt(x ** 2 + y ** 2)
        weight = (1 - math.e ** (-1 * r ** 2 / (2 * self.SIGMA ** 2))) / (self.SIGMA * math.sqrt(2 * math.pi))

        return weight

    def generate(self, shape: tuple) -> np.ndarray:
        h, w = shape

        matrix = np.zeros(shape=(h, w, 3)).astype(np.float32)
        for y in range(h):
            for x in range(w):
                x_n, y_n = self.__normalize(x, y, w, h)
                matrix[y][x] = self.__compute_pixel(x_n, y_n)

        return matrix