from lab2 import *
import numpy as np
import matplotlib.pyplot as plt


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
        arr = (arr).astype(np.float64)

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
        img.color_model = 4
        return img


class Image(GrayScaleTransform):
    # klasa stanowiaca glowny interfejs biblioteki, w pozniejszym czasie bedzie dziedziczyla po kolejnych klasach realizujacych kolejne metody przetwarzania obrazow
    def __init__(self, path=None, data=None) -> None:
        super().__init__(path, data)


# inicjalizator
img = GrayScaleTransform('./lena.png')

# to_gray()
img_gray = img.to_gray()
# plt.gray()
# img_gray.show_img()

# to_sepia()
img_sepia1 = img.to_sepia((1.1, 0.9))
img_sepia2 = img.to_sepia((1.5, 0.5))
img_sepia3 = img.to_sepia((1.9, 0.1))

img_sepia4 = img.to_sepia(w=20)
img_sepia5 = img.to_sepia(w=30)
img_sepia6 = img.to_sepia(w=40)

# figure, axis = plt.subplots(2, 3)
# axis[0, 0].imshow(img_sepia1.data)
# axis[0, 1].imshow(img_sepia2.data)
# axis[0, 2].imshow(img_sepia3.data)
# axis[1, 0].imshow(img_sepia4.data)
# axis[1, 1].imshow(img_sepia5.data)
# axis[1, 2].imshow(img_sepia6.data)
# plt.show()
