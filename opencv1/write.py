import cv2
#cv2 -opencv library
#imread -reads image from file


img = cv2.imread('image1.jpeg',1)


if img is None:
    print("Image not loaded")
else:
    print("Image Loaded")
    cv2.imshow('Window Title',img)
    cv2.waitKey(0) # 0 - waits for a key press and 2000 - waits for 2 seconds
    cv2.destroyAllWindows() # destroys all the windows and frees the memory
    cv2.imwrite('output1.jpeg',img)# writes image to file
    