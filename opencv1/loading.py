import cv2
#cv2 -opencv library
#imread -reads image from file
#1 -flag for color image
#2 -flag for grayscale image
#3 -flag for any type of image
img = cv2.imread('image1.jpeg')


if img is None:
    print("Image not loaded")
else:
    print("Image Loaded")