import numpy as np

from transform.distortion import ImageDistortion

class BBox():
    def __init__(self, x: float, y: float, w: float, h: float) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __clip(self, p: int, s: int) -> tuple[int, int]:
        p = 0 if p < 0 else p
        p = p - 1 if p >= s else p

        return p

    def convert_2_pixel(self, w: int, h: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        l = int(round((self.x - self.w / 2) * w))
        r = int(round((self.x + self.w / 2) * w))
        t = int(round((self.y - self.h / 2) * h))
        b = int(round((self.y + self.h / 2) * h))

        l = self.__clip(l, w)
        r = self.__clip(r, w)
        t = self.__clip(t, h)
        b = self.__clip(b, h)

        return (l, t), (l, b), (r, t), (r, b)
    
    def correct(self, k: float, w: int, h: int) -> tuple[int, int, int, int]:
        bbox = self.convert_2_pixel(w, h)
        distortion = ImageDistortion()
        points = []
        for box in bbox:
            y, x = distortion.transform(box[1], box[0], h, w, k)
            x = self.__clip(x, w)
            y = self.__clip(y, h)
            points.append((x, y))
        points = np.array(points)
        l = min(points[:, 0])
        r = max(points[:, 0])
        t = min(points[:, 1])
        b = max(points[:, 1])

        return l, r, t, b
    
def convert_pixel_2_yolo(l: int, r: int, t: int, b: int, w: int, h: int) -> tuple[float, float, float, float]:
    x = (l + (r - l) / 2) / w
    y = (t + (b - t) / 2) / h
    w = (r - l) / w
    h = (b - t) / h

    return x, y, w, h