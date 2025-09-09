#removing small dots froman image - blurring
#Old scanned documents or security camera footage often have random black & white dots. Median blur removes these without destroying edges. Sorting is non linear operation


# sort the kernel ,replace value with sorted middle value
import cv2

img=cv2.imread('image1.jpeg')
#cv2.medianBlur(image,ksize)
blurred=cv2.medianBlur(img,21)
cv2.imshow("original",img)  
cv2.imshow("blurred",blurred)  
cv2.waitKey(0)
cv2.destroyAllWindows()