import matplotlib.pyplot as plt
import numpy as np
import copy as cp
from enum import Enum
from matplotlib.image import imsave, imread
from math import degrees, sqrt, cos, acos


class ColorModel(Enum):
    rgb = 0
    hsv = 1
    hsi = 2
    hsl = 3
    gray = 4  # obraz 2d


class BaseImage:
    data: np.ndarray  # tensor przechowujacy piksele obrazu
    color_model: ColorModel  # atrybut przechowujacy biezacy model barw obrazu

    def __init__(self, path: str = None, data: np.ndarray = None) -> None:
        # inicjalizator wczytujacy obraz do atrybutu data na podstawie sciezki
        if path != None:
            self.data = imread(path)
            self.data = (self.data * 255).astype(np.uint8)
        else:
            self.data = data

    def save_img(self, path: str) -> None:
        # metoda zapisujaca obraz znajdujacy sie w atrybucie data do pliku
        imsave(path, self.data)

    def show_img(self) -> None:
        # metoda wyswietlajaca obraz znajdujacy sie w atrybucie data
        plt.imshow(self.data)
        plt.show()

    def get_layer(self, layer_id: int) -> 'BaseImage':
        # metoda zwracajaca warstwe o wskazanym indeksie
        layer = cp.deepcopy(self)
        layer.data = self.data[:, :, layer_id]
        return layer

    def to_hsv(self) -> 'BaseImage':
        # metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsv
        # metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        img = cp.deepcopy(self)
        img.data = (img.data).astype(np.float64)

        for row in img.data:
            for col in row:
                r, g, b = col[0], col[1], col[2]
                M = max(r, g, b)
                m = min(r, g, b)

                # r => h
                if g >= b:
                    x = (r - 0.5 * g - 0.5 * b) / \
                        sqrt(r ** 2 + g ** 2 + b ** 2 - r*g - r*b - g*b)

                    col[0] = degrees(acos(x))
                else:
                    x = (r - 0.5 * g - 0.5 * b) / \
                        sqrt(r ** 2 + g ** 2 + b ** 2 - r*g - r*b - g*b)

                    col[0] = 360 - degrees(acos(x))

                # g => s
                if M > 0:
                    col[1] = (1 - (m/M))
                else:
                    col[1] = 0

                # b => v
                col[2] = (M / 255)

        img.color_model = 1
        return img

    def to_hsi(self) -> 'BaseImage':
        # metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsi
        # metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        img = cp.deepcopy(self)
        img.data = img.data.astype(np.float64)

        for row in img.data:
            for col in row:
                r, g, b = col[0], col[1], col[2]
                M = max(r, g, b)
                m = min(r, g, b)

                # r => h
                if g >= b:
                    x = (r - 0.5 * g - 0.5 * b) / \
                        sqrt(r ** 2 + g ** 2 + b ** 2 - r * g - r * b - g * b)

                    col[0] = degrees(acos(x))
                else:
                    x = (r - 0.5 * g - 0.5 * b) / \
                        sqrt(r ** 2 + g ** 2 + b ** 2 - r * g - r * b - g * b)

                    col[0] = 360 - degrees(acos(x))

                # b => i
                col[2] = (r + g + b) / 3

                # g => s
                if col[2] > 0:
                    col[1] = (1 - (m/col[2]))
                else:
                    col[1] = 0

        img.color_model = 2
        return img

    def to_hsl(self) -> 'BaseImage':
        # metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsl
        # metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        img = cp.deepcopy(self)
        img.data = img.data.astype(np.float64)

        for row in img.data:
            for col in row:
                r, g, b = col[0], col[1], col[2]
                M = max(r, g, b)
                m = min(r, g, b)
                d = (M - m) / 255

                # r => h
                if g >= b:
                    x = (r - 0.5 * g - 0.5 * b) / \
                        sqrt(r ** 2 + g ** 2 + b ** 2 - r * g - r * b - g * b)

                    col[0] = degrees(acos(x))
                else:
                    x = (r - 0.5 * g - 0.5 * b) / \
                        sqrt(r ** 2 + g ** 2 + b ** 2 - r * g - r * b - g * b)

                    col[0] = 360 - degrees(acos(x))

                # b => l
                l = (0.5 * (M + m)) / 255
                col[2] = l

                # g => s
                if l > 0:
                    col[1] = (d / (1 - abs((2 * l) - 1)))
                else:
                    col[1] = 0

        img.color_model = 3
        return img

    def to_rgb(self) -> 'BaseImage':
        # metoda dokonujaca konwersji obrazu w atrybucie data do modelu rgb
        # metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        img = cp.deepcopy(self)

        for row in img.data:
            for col in row:
                # hsv => rgb
                if img.color_model == 1:
                    h, s, v = col[0], col[1], col[2]
                    M = v * 255
                    m = M * (1 - s)
                    z = (M - m) * (1 - abs(((h / 60) % 2) - 1))

                    if h >= 0 and h < 60:
                        col[0] = M
                        col[1] = z + m
                        col[2] = m
                    elif h >= 60 and h < 120:
                        col[0] = z + M
                        col[1] = M
                        col[2] = m
                    elif h >= 120 and h < 180:
                        col[0] = m
                        col[1] = M
                        col[2] = z + m
                    elif h >= 180 and h < 240:
                        col[0] = m
                        col[1] = M
                        col[2] = z + m
                    elif h >= 240 and h < 300:
                        col[0] = z + m
                        col[1] = m
                        col[2] = M
                    elif h >= 300 and h < 360:
                        col[0] = M
                        col[1] = m
                        col[2] = z + m

                # hsi => rgb
                elif img.color_model == 2:
                    h, s, i = col[0], col[1], col[2]

                    if h == 0:
                        col[0] = i + (2 * i * s)
                        col[1] = i - (i * s)
                        col[2] = i - (i * s)
                    elif h > 0 and h < 120:
                        col[0] = i + (i * s * cos(h) / cos(60 - h))
                        col[1] = i + (i * s * (1 - cos(h) / cos(60 - h)))
                        col[2] = i - (i * s)
                    elif h == 120:
                        col[0] = i - (i * s)
                        col[1] = i + (2 * i * s)
                        col[2] = i - (i * s)
                    elif h > 120 and h < 240:
                        col[0] = i - (i * s)
                        col[1] = i + (i * s * cos(h - 120) / cos(180 - h))
                        col[2] = i + (i * s *
                                      (1 - cos(h - 120) / cos(180 - h)))
                    elif h == 240:
                        col[0] = i - (i * s)
                        col[1] = i - (i * s)
                        col[2] = i + (2 * i * s)
                    elif h > 240 and h < 360:
                        col[0] = i + (i * s *
                                      (1 - cos(h - 240) / cos(300 - h)))
                        col[1] = i - (i * s)
                        col[2] = i + (i * s * cos(h - 240) / cos(300 - h))

                # hsl => rgb
                if img.color_model == 3:
                    h, s, l = col[0], col[1], col[2]
                    d = s * (1 - abs((2 * l) - 1))
                    m = 255 * (l - (0.5 * d))
                    x = 255 * d * (1 - abs(((h / 60) % 2) - 1))
                    d *= 255

                    if h >= 0 and h < 60:
                        col[0] = d + m
                        col[1] = x + m
                        col[2] = m
                    elif h >= 60 and h < 120:
                        col[0] = x + m
                        col[1] = d + m
                        col[2] = m
                    elif h >= 120 and h < 180:
                        col[0] = m
                        col[1] = d + m
                        col[2] = x + m
                    elif h >= 180 and h < 240:
                        col[0] = m
                        col[1] = x + m
                        col[2] = d + m
                    elif h >= 240 and h < 300:
                        col[0] = x + m
                        col[1] = m
                        col[2] = d + m
                    elif h >= 300 and h < 360:
                        col[0] = d + m
                        col[1] = m
                        col[2] = x + m

        img.data = (img.data).astype(np.uint8)
        img.color_model = 0
        return img


# inicjalizator
img = BaseImage('./lena.png')
img.color_model = 0

# show_img()
# img.show_img()

# save_img()
# img.save_img('./copy.png')

# get_layer()
r = img.get_layer(0).data
g = img.get_layer(1).data
b = img.get_layer(2).data
# plt.imshow(np.dstack((r, g, b)))
# plt.show()

# print("RGB:", img.data[0:1])
# print()

# # to_hsv()
img_hsv = img.to_hsv()
# print("HSV:", img_hsv.data[0:1])
# print()

# # to_hsi()
img_hsi = img.to_hsi()
# print("HSI:", img_hsi.data[0:1])
# print()

# # to_hsl()
img_hsl = img.to_hsl()
# print("HSL:", img_hsl.data[0:1])
# print()

# # to_rgb()
from_hsv = img_hsv.to_rgb()
from_hsi = img_hsi.to_rgb()
from_hsl = img_hsl.to_rgb()
# print(from_hsl.data[0:1])

# fig, ax = plt.subplots(2, 2)
# ax[0, 0].imshow(img.data)
# ax[0, 0].set_title("RGB")
# ax[0, 1].imshow(from_hsv.data)
# ax[0, 1].set_title("HSV to RGB")
# ax[1, 0].imshow(from_hsi.data)
# ax[1, 0].set_title("HSI to RGB")
# ax[1, 1].imshow(from_hsl.data)
# ax[1, 1].set_title("HSL to RGB")
# plt.show()
