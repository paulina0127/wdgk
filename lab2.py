from BaseImage import *
import matplotlib.pyplot as plt

# inicjalizator
img = BaseImage('./img/lena.png')
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
# print(from_hsi.data[0:1])

fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(img.data)
ax[0, 0].set_title("RGB")
ax[0, 1].imshow(from_hsv.data)
ax[0, 1].set_title("HSV to RGB")
ax[1, 0].imshow(from_hsi.data)
ax[1, 0].set_title("HSI to RGB")
ax[1, 1].imshow(from_hsl.data)
ax[1, 1].set_title("HSL to RGB")
plt.show()
