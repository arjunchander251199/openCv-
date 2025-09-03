#cropping matrix
import cv2

img = cv2.imread('image1.jpeg')

if img is None:
    print('image not found')
else:
    print("image loaded successfully")
    cropped = img[0:500, 100:500]   #starty :endy, startx:endx
    cv2.imshow('original image window', img)
    cv2.imshow('cropped image window', cropped)

    cv2.imwrite('cropped.jpeg', cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 





