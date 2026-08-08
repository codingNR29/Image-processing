import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

im = cv.imread('', cv.IMREAD_GRAYSCALE)

im_lpf = cv.GaussianBlur(im, (51, 51), 20)

im_hpf = cv.subtract(im, im_lpf)

im_high_con = cv.addWeighted(im, 1.0, im_hpf,  1.5, 0)

fig, ax = plt.subplots(2, 2, figsize=(50, 300))
ax[0, 0].imshow(im, cmap='gray')
ax[0, 0].set_title('Original Image')

ax[0, 1].imshow(im_lpf, cmap='gray')
ax[0, 1].set_title('Low Pass Filtered Image')

ax[1, 0].imshow(im_hpf, cmap='gray')
ax[1, 0].set_title('High Pass Filtered Image')
ax[1, 1].imshow(im_high_con, cmap='gray')
ax[1, 1].set_title('High Contrast Image')

for a in ax.flatten():
    a.set_xticks([])
    a.set_yticks([])

plt.show()
