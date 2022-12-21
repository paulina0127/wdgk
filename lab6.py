from BaseImage import *
from Histogram import *
from Image import *
import matplotlib.pyplot as plt
import numpy as np

# inicjalizator
img = Image('./img/lena.png')

# conv_2d()
# filtr górnoprzepustowy - wyostrzenie
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# img_filtered = img.conv_2d(kernel)

# filtr dolnoprzepustowy - rozmycie
kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
# img_filtered = img.conv_2d(kernel, 1/9)

# rozmycie gaussowskie
kernel = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [
                  6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])
img_filtered = img.conv_2d(kernel, 1/256)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img.data)
ax[1].imshow(img_filtered.data)
plt.show()
