from BaseImage import *
from Histogram import *
from Image import *

# inicjalizator
img = Image('./img/lena.png')
hist = Histogram(img.data)

gray = img.to_gray()
hist_gray = Histogram(gray.data)

# plot()
# hist.plot()
hist_gray.plot()

# compare_to()
img1 = Image('./img/lena.png')
img1.data = img1.to_gray().data

img2 = Image('./img/lena1.png')
img2.data = img2.to_gray().data

print(img1.compare_to(img1, ImageDiffMethod(0)))
print(img1.compare_to(img1, ImageDiffMethod(1)))
print()

print(img1.compare_to(img2, ImageDiffMethod(0)))
print(img1.compare_to(img2, ImageDiffMethod(1)))

# hist_cum = hist_gray.to_cumulated()

# bins = np.array(range(256))
# fig, ax = plt.subplots(1, 2)
# ax[0].plot(bins, hist_gray.values[0], color="gray")
# ax[1].plot(bins, hist_cum.values[0], color="gray")
# plt.show()
