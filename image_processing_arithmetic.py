import cv2 as cv
import numpy as np

x = np.uint8([250])
y = np.uint8([10])

result = cv.add(x,y)
print(result)

print(x+y)


# image blending
img1 = cv.imread(cv.samples.findFile("../face-recognition/face-db/j2-blue.png"))
img2 = cv.imread(cv.samples.findFile("../face-recognition/face-db/jenkins.jpg"))

blend = cv.addWeighted(img1,0.7,img2,0.3,0)

cv.imshow('test blend', blend)
cv.waitKey(0)
cv.destroyAllWindows()

# bitwise operation
# add logo in top left corner so create ROI
rows, cols, channel = img2.shape
roi = img1[0:rows,0:cols]

# create mask of logo and create its inverse mask also
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)

# blackout area of logo ROI
img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
# take only region of logo
img2_fg = cv.bitwise_and(img2, img2, mask = mask)
# put logo in ROI and modify main img
dst = cv.add(img1_bg,img2_fg)
img1[0:rows,0:cols] =  dst
cv.imshow('res',img1)
cv.waitKey(0)
cv.destroyAllWindows()