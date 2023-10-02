import math
import numpy as np

class ImageDistortion():
    def __normalize(self, p: int, s: int, cap: int) -> float:
        return (2 * (p + (cap - s) / 2) - cap + 1)/(cap - 1)
    
    def __denormalize(self, p: float, s: int, cap: int) -> int:
        return int(round((p + (s - 1) / (cap - 1)) * (cap - 1) / 2))

    def __compute_distorted_pixel(self, p: float, r: float, k: float) -> float:
        distorted_r = (1 - math.sqrt(1 - 4 * k * r ** 2)) / (2 * k * r)
        return p * distorted_r / r

    def __distort(self, x: float, y: float, r: float, k: float) -> tuple[float, float]:
        if 4 * k * r ** 2 > 1:
            return 2, 2
        if r == 0 or k == 0:
            return x, y
        return self.__compute_distorted_pixel(x, r, k), self.__compute_distorted_pixel(y, r, k)
    
    def transform(self, x: int, y: int, h: float, w: float, k: int) -> tuple[int, int]:
        cap = max(h, w)
        x_norm, y_norm = self.__normalize(x, h, cap), self.__normalize(y, w, cap)
        r_u = math.sqrt(x_norm ** 2 + y_norm ** 2)
        x_d, y_d = self.__distort(x_norm, y_norm, r_u, k)
        x_u, y_u = self.__denormalize(x_d, h, cap), self.__denormalize(y_d, w, cap)
        return x_u, y_u
    
    def distort(self, image: np.ndarray, k: float) -> np.ndarray:
        ''' image: format (h, w, c) with c = 3 (R, G, B)
        k: distortion parameter
        '''
        h, w = float(image.shape[0]), float(image.shape[1])
        distorted_image = np.zeros_like(image)

        for x in range(distorted_image.shape[0]):
            for y in range(distorted_image.shape[1]):
                x_u, y_u = self.transform(x, y, h, w, k)
                if 0 <= x_u and x_u < h and 0 <= y_u and y_u < w:
                    distorted_image[x][y] = image[x_u][y_u]

        return distorted_image.astype(np.uint8)