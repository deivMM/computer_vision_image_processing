import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

image = cv2.imread('../data/images/coins.jpg')
image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
image = cv2.GaussianBlur(image, (5, 5), 5)

brg_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

hist_2d = cv2.calcHist([hsv_image], [0, 1], None, [180, 256], [0, 180, 0, 256])

lowerBound = np.array([55, 65, 0])
upperBound = np.array([80, 130 ,255])
mask = cv2.inRange(hsv_image, lowerBound, upperBound)

f, ax = plt.subplots(3, 1, figsize=(5, 8))

ax[0].imshow(brg_image)
ax[0].axis('off')

divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)

im = ax[1].imshow(hist_2d, cmap='jet')
ax[1].set_ylabel('Hue')
ax[1].set_xlabel('Saturation')

ax[2].imshow(mask, cmap='gray')

plt.colorbar(im, cax=cax, ticks=[])
plt.subplots_adjust(hspace=0.5)
plt.show()

