import cv2

img = cv2.imread('image1.jpeg')

text = "Hello World"

font = cv2.FONT_HERSHEY_SIMPLEX
size = 10
position = (100,800) #(x,y) bottom left corner
color = (255,0,0) #BGR
thickness = 4

if img is not None:
    cv2.putText(img,text,position,font,size,color,thickness)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()