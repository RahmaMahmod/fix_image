import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFilter, ImageEnhance
import cv2
from cv2 import dnn_superres

img = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)
print(img.shape)

_, mask = cv2.threshold(img, 15, 50, cv2.THRESH_BINARY_INV)

kernal = np.ones((3, 3), np.uint8)

dilation = cv2.dilate(mask, kernal, iterations=3)

maintained_img = cv2.inpaint(img, dilation, 10, cv2.INPAINT_TELEA)

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

print(maintained_img.shape)
# Read the desired model
path = "ESPCN_x4.pb"
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("espcn", 4)

# Upscale the image
upsample_img = sr.upsample(maintained_img)
print(upsample_img.shape)

im = Image.fromarray(upsample_img)
im.save("ESPCN_img.jpeg")

#make background black
ret, thresh4 = cv2.threshold(upsample_img, 30, 55,cv2.THRESH_TOZERO)

blackbackground_img = Image.fromarray(thresh4)
blackbackground_img.save("blackbackground_img.jpeg")

titles = ['image', 'maintained_img', 'upsample_img','thresh4']
images = [img, maintained_img, upsample_img, thresh4]

for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()


