import cv2 as cv
import sys

img = cv.imread(cv.samples.findFile("../face-recognition/face-db/jenkins.jpg"))

if img is None:
    sys.exit("gabisa get image!")

cv.imshow("show image : ",img)
k = cv.waitKey(0)

if(k==ord('s')):
    cv.imwrite("../face-recognition/face-db/jenkins.png",img)