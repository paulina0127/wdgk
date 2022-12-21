from BaseImage import *
from Histogram import *
import numpy as np
import copy as cp
from enum import Enum
from math import sqrt
from typing import Optional


class GrayScaleTransform(BaseImage):
    def __init__(self, path=None, data=None) -> None:
        super().__init__(path, data)

    def to_gray(self) -> BaseImage:
        # metoda zwracajaca obraz w skali szarosci jako obiekt klasy BaseImage
        r, g, b = self.data[:, :, 0], self.data[:, :, 1], self.data[:, :, 2]
        arr = 0.2989 * r + 0.5870 * g + 0.1140 * b

        arr = (arr).astype(np.uint8)
        img = BaseImage(data=arr)
        img.color_model = 4
        return img

    def to_sepia(self, alpha_beta: tuple = (None, None), w: int = None) -> BaseImage:
        # metoda zwracajaca obraz w sepii jako obiekt klasy BaseImage
        # sepia tworzona metoda 1 w przypadku przekazania argumentu alpha_beta lub metoda 2 w przypadku przekazania argumentu w
        self = self.to_gray()
        arr = np.dstack((self.data, self.data, self.data))
        arr = (arr).astype(np.float32)

        if (alpha_beta != (None, None) and alpha_beta[0] > 1 and alpha_beta[1] < 1 and sum(alpha_beta) == 2):
            arr[:, :, 0] *= alpha_beta[0]
            arr[:, :, 2] *= alpha_beta[1]
        elif (w != None and w >= 20 and w <= 40):
            arr[:, :, 0] += 2 * w
            arr[:, :, 1] += w

        arr[arr > 255] = 255
        arr[arr < 0] = 0
        arr = (arr).astype(np.uint8)
        img = BaseImage(data=arr)
        img.color_model = 5
        return img


class ImageDiffMethod(Enum):
    mse = 0
    rmse = 1


class Image(GrayScaleTransform):
    def __init__(self, path=None, data=None) -> None:
        super().__init__(path, data)


class ImageComparison(BaseImage):
    # klasa reprezentujaca obraz, jego histogram oraz metody porównania
    def __init__(self, path=None, data=None) -> None:
        super().__init__(path, data)

    def histogram(self) -> Histogram:
        # metoda zwracajaca obiekt zawierajacy histogram biezacego obrazu (1- lub wielowarstwowy)
        return Histogram(self.data)

    def compare_to(self, other: Image, method: ImageDiffMethod) -> float:
        # metoda zwracajaca mse lub rmse dla dwoch obrazow
        x = self.histogram().values[0]
        y = other.histogram().values[0]

        result = np.sum(np.subtract(x, y) ** 2) / 256

        if method.name == "mse":
            return result
        elif method.name == "rmse":
            return sqrt(result)


class ImageAligning(BaseImage):
    # klasa odpowiadająca za wyrównywanie hostogramu
    def __init__(self, path=None, data=None) -> None:
        super().__init__(path, data)

    def align_image(self, tail_elimination: bool = True) -> Image:
        # metoda zwracajaca poprawiony obraz metoda wyrownywania histogramow
        img = cp.deepcopy(self)
        img.data = (img.data).astype(np.float16)

        if img.data.ndim == 2:
            dim = 1
        else:
            dim = 3

        for d in range(dim):
            layer = img.get_layer(d).data

            if tail_elimination == False:
                min, max = np.min(layer), np.max(layer)
            else:
                hist_cum = Histogram(layer).to_cumulated().values[0]
                x = hist_cum[-1]

                min = hist_cum.index([i for i in hist_cum if i > x * 0.05][0])
                max = hist_cum.index(
                    [i for i in hist_cum if i > x - (x * 0.05)][0])

            for row in range(layer.shape[0]):
                for col in range(layer.shape[1]):
                    layer[row, col] = (
                        layer[row, col] - min) * (255 / (max - min))

            if dim == 3:
                img.data[:, :, d] = layer
            else:
                img.data = layer

        img.data[img.data > 255] = 255
        img.data[img.data < 0] = 0
        img.data = (img.data).astype(np.uint8)
        return img


# funkcja pomocnicza dla conv_2d
def conv(arr: np.ndarray, kernel: np.ndarray) -> np.float16:
    sum = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            sum += arr[i, j] * kernel[i, j]

    sum = (sum).astype(np.float16)
    return sum


class ImageFiltration:
    def conv_2d(self, kernel: np.ndarray, prefix: Optional[float] = None) -> Image:
        # kernel: filtr w postaci tablicy numpy
        # prefix: przedrostek filtra, o ile istnieje; Optional - forma poprawna obiektowo, lub domyslna wartosc = 1 - optymalne arytmetycznie
        img = cp.deepcopy(self)
        img.data = img.data.astype(np.float16)
        shape = kernel.shape[0]
        row = col = 0

        if (prefix):
            kernel = prefix * kernel

        if img.data.ndim == 2:
            dim = 1
        else:
            dim = 3

        for d in range(dim):
            layer = img.get_layer(d).data
            filtr = cp.deepcopy(layer)

            for i in range(layer.shape[0] - 2):
                for j in range(layer.shape[1] - 2):
                    arr = layer[row:row + shape, col:col + shape]
                    filtr[row + (shape // 2), col + (shape //
                                                     2)] = conv(arr, kernel)
                    col += 1

                row += 1
                col = 0
            row = col = 0

            if dim == 3:
                img.data[:, :, d] = filtr
            else:
                img.data = filtr

        img.data[img.data > 255] = 255
        img.data[img.data < 0] = 0
        img.data = (img.data).astype(np.uint8)
        return img


class Image(GrayScaleTransform, ImageAligning, ImageComparison, ImageFiltration):
    def __init__(self, path=None, data=None) -> None:
        super().__init__(path, data)
