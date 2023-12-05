import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread(cv.samples.findFile("../face-recognition/face-db/jenkins.jpg"))
assert img is not None, "file not found! "

px = img[100,100]
print(px)


# accessing blue pixel
blue = img[100,100,0]
print(blue)

# modify pixel
# img[100,100] = [255,255,255]
# print(img[100,100])


# accesing red value
img.item(10,10,2)
# print(img)

# modify red value
img.itemset((10,10,2),100)
img.item(10,10,2)

# accessing image properties
print(img.shape)
print(img.size)
print(img.dtype)

# split to red green blue channel
b,g,r = cv.split(img)

# merge rgb channel
img_merge = cv.merge((b,g,r))

cv.imwrite("../face-recognition/face-db/jenkins-edit.png",img)
cv.imwrite("../face-recognition/face-db/j2-red.png",r)
cv.imwrite("../face-recognition/face-db/j2-blue.png",b)
cv.imwrite("../face-recognition/face-db/j2-green.png",g)
cv.imwrite("../face-recognition/face-db/jenkins-merge.png",img_merge)

BLUE = [255,0,0]




# border

replicate = cv.copyMakeBorder(img, 5,5,5,5,cv.BORDER_REPLICATE)
reflect = cv.copyMakeBorder(img, 5,5,5,5,cv.BORDER_REFLECT)
reflect_101 = cv.copyMakeBorder(img, 5,5,5,5,cv.BORDER_REFLECT_101)
reflect101 = cv.copyMakeBorder(img, 5,5,5,5,cv.BORDER_REFLECT101)
wrap = cv.copyMakeBorder(img, 5,5,5,5,cv.BORDER_WRAP)
constant = cv.copyMakeBorder(img, 5,5,5,5,cv.BORDER_CONSTANT, value= BLUE)

plt.subplot(231),plt.imshow(img,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

plt.show()