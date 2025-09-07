import cv2

img = cv2.imread('image1.jpeg')

#(x1,y1) is the starting point (top left corner)
pt1=(100,100)

#(x2,y2) is the end point (bottom right corner)
pt2=(1000,1000)

color = (255,0,0) #BGR
thickness = 2 #-1 for filling the rectangle

if img is not None:
    cv2.rectangle(img,pt1,pt2,color,thickness)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  

else:
    print("No image")