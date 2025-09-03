#shape attribute of image
#height and width
#channels

import cv2
#cv2 -opencv library
#imread -reads image from file


img = cv2.imread('image1.jpeg',1)


if img is not None:
    h,w,c=img.shape
    print("Image Loaded")
    print("Height = ",h)
    print("Width = ",w)
    print("Channels = ",c)
    cv2.imshow('Window Title',img)
    cv2.waitKey(0) # 0 - waits for a key press and 2000 - waits for 2 seconds
    cv2.destroyAllWindows() # destroys all the windows and frees the memory
else:
    print("Image not loaded")


#if return 1 or no channel then its grayscale