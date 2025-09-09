#sharpening - makes images crisp and clear. Highlights the edges of an image and makes them sharper.
#our of focus images sharpened
#sharpening makes the edges of an image clearer and crisper 


#cv2.filter2d(image,depth,kernel)

import cv2
import numpy as np

# Read the image
img = cv2.imread('image1.jpeg')

# Define sharpening kernel
kernel = np.array([[ 0, -1,  0],
                   [-1,  8, -1],
                   [ 0, -1,  0]])

# Apply the kernel to the image
#-1 number of bits per pixel - intensity of each pixel -wavelength

sharpened = cv2.filter2D(img, -1, kernel)

# Show original and sharpened images
cv2.imshow("Original", img)
cv2.imshow("Sharpened", sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
