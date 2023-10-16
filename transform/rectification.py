import cv2
import math
import numpy as np
from .distortion import ImageDistortion

class ImageRectification(ImageDistortion):
    def __compute_rectified_pixel(self, p: float, r: float, k: float) -> float:
        rectified_r = r / (1 + k * r ** 2)
        return p * rectified_r / r
    
    def __rectify(self, x: float, y: float, r: float, k: float) -> tuple[float, float]:
        return self.__compute_rectified_pixel(x, r, k), self.__compute_rectified_pixel(y, r, k)
    
    def __sharpen(self, image: np.ndarray) -> np.ndarray:
        kernel = np.array([[-0.5, 0.5, -0.5],
                           [0.5, 1, 0.5], 
                           [-0.5, 0.5, -0.5]])
        
        return cv2.filter2D(image, -1, kernel)
    
    def transform(self, x: int, y: int, h: float, w: float, k: int) -> tuple[int, int]:
        cap = max(h, w)
        x_norm, y_norm = self._normalize(x, h, cap), self._normalize(y, w, cap)
        r_u = math.sqrt(x_norm ** 2 + y_norm ** 2)
        x_d, y_d = self.__rectify(x_norm, y_norm, r_u, k)
        x_u, y_u = self._denormalize(x_d, h, cap), self._denormalize(y_d, w, cap)
        return x_u, y_u
    
    def rectify(self, image: np.ndarray, k: float) -> np.ndarray:
        ''' image: format (h, w, c) with c = 3 (R, G, B)
        k: distortion parameter
        '''
        h, w = float(image.shape[0]), float(image.shape[1])
        rectified_image = np.zeros_like(image)

        for x in range(rectified_image.shape[0]):
            for y in range(rectified_image.shape[1]):
                try:
                    x_u, y_u = self.transform(x, y, h, w, k)
                    if 0 <= x_u and x_u < h and 0 <= y_u and y_u < w:
                        rectified_image[x][y] = image[x_u][y_u]
                except:
                    continue

        return self.__sharpen(cv2.medianBlur(rectified_image.astype(np.uint8), 3))