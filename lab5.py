from BaseImage import *
from Histogram import *
from Image import *
import matplotlib.pyplot as plt
import numpy as np

# inicjalizator
img = Image('./img/lena.png')

gray = img.to_gray()
gray = Image(data=gray.data)

# align_image()
gray_align = gray.align_image(tail_elimination=False)
gray_align_tail = gray.align_image(tail_elimination=True)

# hist_gray = gray.histogram()
# hist_gray_cum = hist_gray.to_cumulated()
# hist_gray_align = gray_align.histogram()
# hist_gray_align_tail = gray_align_tail.histogram()

# bins = np.array(range(256))
# fig, ax = plt.subplots(2, 2)
# ax[0, 0].plot(bins, hist_gray.values[0], color="gray")
# ax[0, 0].set_title("Histogram")
# ax[0, 1].plot(bins, hist_gray_cum.values[0], color="gray")
# ax[0, 1].set_title("Histogram skumulowany")
# ax[1, 0].plot(bins, hist_gray_align.values[0], color="gray")
# ax[1, 0].set_title("Histogram bez eliminacji ogonów")
# ax[1, 1].plot(bins, hist_gray_align_tail.values[0], color="gray")
# ax[1, 1].set_title("Histogram z eliminacją ogonów")
# plt.show()

fig, ax = plt.subplots(1, 3)
ax[0].imshow(gray.data, cmap="gray")
ax[0].set_title("Obraz")
ax[1].imshow(gray_align.data, cmap="gray")
ax[1].set_title("Obraz po wyrównaniu histogramu")
ax[2].imshow(gray_align_tail.data, cmap="gray")
ax[2].set_title("Obraz po wyrównaniu histogramu z eliminacją ogonów")
plt.show()

##############################################################################

lungs = Image('./img/lungs.jpg')
lungs_align = lungs.align_image(tail_elimination=False)
lungs_align_tail = lungs.align_image(tail_elimination=True)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(lungs.data, cmap="gray")
ax[0].set_title("Obraz")
ax[1].imshow(lungs_align.data, cmap="gray")
ax[1].set_title("Obraz po wyrównaniu histogramu")
ax[2].imshow(lungs_align_tail.data, cmap="gray")
ax[2].set_title("Obraz po wyrównaniu histogramu z eliminacją ogonów")
plt.show()

##############################################################################

img_align = img.align_image(tail_elimination=False)
img_align_tail = img.align_image(tail_elimination=True)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(img.data)
ax[0].set_title("Obraz")
ax[1].imshow(img_align.data)
ax[1].set_title("Obraz po wyrównaniu histogramu")
ax[2].imshow(img_align_tail.data)
ax[2].set_title("Obraz po wyrównaniu histogramu z eliminacją ogonów")
plt.show()
