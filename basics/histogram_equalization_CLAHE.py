import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

image = cv2.imread('../data/images/coins.jpg')
image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

equImg = cv2.equalizeHist(gray_image)
equhist = cv2.calcHist([equImg], [0], None, [256], [0, 256])
equcdf = equhist.cumsum()
equcdf_normalized = equcdf * equhist.max() / equcdf.max()

claheObj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
claheImg = claheObj.apply(gray_image)
clahehist = cv2.calcHist([claheImg], [0], None, [256], [0, 256])
clahecdf = clahehist.cumsum()
clahecdf_normalized = clahecdf * clahehist.max() / clahecdf.max()

f, ax = plt.subplots(2, 3, figsize=(10, 5))

ax[0,0].imshow(gray_image, cmap='gray')
ax[1,0].plot(hist, color='b')
ax[1,0].plot(cdf_normalized, color='k')

ax[0,1].imshow(equImg, cmap='gray')
ax[1,1].plot(hist, color='b')
ax[1,1].plot(equcdf_normalized, color='k')

ax[0,2].imshow(claheImg, cmap='gray')
ax[1,2].plot(clahehist, color='b')
ax[1,2].plot(clahecdf_normalized, color='k')

ax[0,0].axis('off')
ax[0,1].axis('off')
ax[0,2].axis('off')
plt.subplots_adjust(wspace=0.2)
plt.show()
