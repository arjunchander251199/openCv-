import cv2

# Read two images
img1 = cv2.imread('image1.jpeg')
img2 = cv2.imread('flower.jpg')

# Resize both images so they are the same size (bitwise needs same dimensions)
img1 = cv2.resize(img1, (400, 400))
img2 = cv2.resize(img2, (400, 400))

# ------------------------------------------------
# Bitwise operations
# ------------------------------------------------

# #AND → only overlap

# OR → both combined

# XOR → only differences

# NOT → inverted colors

# AND: only keeps the pixels that are bright in BOTH images.
# → Imagine overlap area → only that is visible.
bit_and = cv2.bitwise_and(img1, img2)

# OR: keeps all bright pixels from BOTH images.
# → Looks like both images merged together.
bit_or = cv2.bitwise_or(img1, img2)

# XOR: keeps only the parts that are different between the two.
# → Common/overlap is removed, rest is visible.
bit_xor = cv2.bitwise_xor(img1, img2)

# NOT: inverts the image (like photo negative).
# → Black becomes white, colors flip to their opposites.
bit_not1 = cv2.bitwise_not(img1)  # invert first image
bit_not2 = cv2.bitwise_not(img2)  # invert second image

# ------------------------------------------------
# Show everything
# ------------------------------------------------
cv2.imshow('Image 1 (Original)', img1)
cv2.imshow('Image 2 (Original)', img2)
cv2.imshow('Bitwise AND (only overlap kept)', bit_and)
cv2.imshow('Bitwise OR (merge of both)', bit_or)
cv2.imshow('Bitwise XOR (differences only)', bit_xor)
cv2.imshow('Bitwise NOT - Image1 (inverted)', bit_not1)
cv2.imshow('Bitwise NOT - Image2 (inverted)', bit_not2)

# Wait until a key is pressed, then close windows
cv2.waitKey(0)
cv2.destroyAllWindows()



