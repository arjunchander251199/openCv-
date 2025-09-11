
#binary thresholding - ocr - scanning images
import cv2
#cv2.imread_graschale - another method for converting to grayscale
img = cv2.imread('flower.jpg',cv2.IMREAD_GRAYSCALE)


#cv2.threshold(src,thresh,maxval,type)
#src - source image
#thresh - threshold value
#maxval - maximum value
#type - thresholding type/technique

#less than thresh is black
#more than thresh is white
#more than max becomes 255 automatically

#ret will return the threshold value only
ret ,thresh = cv2.threshold(img,70,255,cv2.THRESH_BINARY)
cv2.imshow('image',img)
cv2.imshow('thresh',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()