import cv2

img = cv2.imread('image1.jpeg')

#(x1,y1) is the starting point
pt1=(500,500)

radius = 300
color = (255,0,0) #BGR
thickness = -1 #-1 for filling the circle

if img is not None:
    cv2.circle(img,pt1,radius,color,thickness)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()