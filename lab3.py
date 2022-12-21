from BaseImage import *
from Histogram import *
from Image import *
import matplotlib.pyplot as plt

# inicjalizator
img = GrayScaleTransform('./img/lena.png')

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

figure, axis = plt.subplots(2, 3)
axis[0, 0].imshow(img_sepia1.data)
axis[0, 1].imshow(img_sepia2.data)
axis[0, 2].imshow(img_sepia3.data)
axis[1, 0].imshow(img_sepia4.data)
axis[1, 1].imshow(img_sepia5.data)
axis[1, 2].imshow(img_sepia6.data)
plt.show()
