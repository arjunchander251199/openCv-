#Canny Edge Detection
#detect borders in an image
#seperate objects
# feature extraction 
#lane detection


import cv2


img = cv2.imread('flower.jpg',2)
#cv2.Canny(img,threshold1,threshold2)
#t1 and t2 are intensity(brightness)
#less than threshold 1 is black (noise)
#greater than threshold 2 is white (strong edge)
#between threshold 1 and 2 is white if its a contected to strong edge or black if its not
edges = cv2.Canny(img,30,60)

cv2.imshow('original',img)
cv2.imshow('edges',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

