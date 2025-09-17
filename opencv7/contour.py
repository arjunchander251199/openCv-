# What are Contours?

# A contour is simply a curve joining all continuous points (having the same color or intensity) along a boundary.

# In other words, Contours = Boundaries of shapes/objects in an image.

# So if you have a flower image:

# Contours will be the green outlines around petals, leaves, etc.

#shape detector 




import cv2

# Read image
img = cv2.imread("donut.jpg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply threshold
ret , thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Find contours
#thresh- binary image without any noise
# RETR_TREE - retrieve all the contours , builds full parent–child hierarchy tree


# With CHAIN_APPROX_NONE

# OpenCV stores every pixel point along the boundary.

# For rectangle →

# Top edge = 100 points

# Bottom edge = 100 points

# Left edge = 50 points

# Right edge = 50 points

# Total ≈ 300+ points



# 👉 Very detailed, big array.


#cv2.CHAIN_APPROX_SIMPLE - only for four points of reactancle
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
#cv2.drawContours(image, contours, contourIndex, color, thickness)


#img= image for which we find contours
#contours - all points found at each level
#index - -1 means all contours , 1 means at index one and so on 
#color -BRG
#thickness - thickness of the contour drawn 
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

# Show image
cv2.imshow("Contours", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(contours)


