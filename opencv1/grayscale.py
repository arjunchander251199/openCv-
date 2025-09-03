#low processing and complexity for grayscale images

import cv2
#cv2 -opencv library
#imread -reads image from file


img = cv2.imread('image1.jpeg',0)
print("Image Loaded")
cv2.imshow('Window Title',img)
cv2.waitKey(0) # 0 - waits for a key press and 2000 - waits for 2 seconds
cv2.destroyAllWindows() # destroys all the windows and frees the memory
cv2.imwrite('output2.jpeg',img)# writes image to file




img2 = cv2.imread('image2.jpeg',1)
print("Image Loaded")
cv2.imshow('Window Title',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
grey=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
cv2.imshow('Window Title',grey)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output3.jpeg',img2)