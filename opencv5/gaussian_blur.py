#potrait image - blur bg image
#make image softer by removing noise
# noise - unwanted information

#red cube in white background where white background has dirt (noise) . Also edges of suqare are blurred a bit for softeneing (less harsh to process)


# A little Gaussian blur → smooths edges and reduces noise (good)

# Too much blur → image becomes very soft and loses all details; edges disappear, small objects vanish, everything looks “mushy”


##
import cv2
#averaging out pixels with neighbours - blurring


img=cv2.imread('image1.jpeg')
#Guassian blur is a linear filter blurs an image by using the Gaussian function- input image -If you double the input image values → output also doubles.

#cv2.GaussianBlur(image,(kernel_sizex,kernel_sizey),sigma)

#kernel x and y size is the size of the kernel - Means number of pixels in width and height - Tells what range(width and height) of pixels will be blurred - region of blur

# sigma tells how much the blur will be - higher value means more blur intensity


blurred=cv2.GaussianBlur(img,(119,119),3)

cv2.imshow("original",img)  
cv2.imshow("blurred",blurred)  
cv2.waitKey(0)

cv2.destroyAllWindows() 