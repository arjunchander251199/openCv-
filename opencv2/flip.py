import cv2

img = cv2.imread('image1.jpeg')
if img is None:
    print('image not found')
else:
    print("image loaded successfully")
#     What the flipCode means:
# 0 = Vertical flip (flip top to bottom)
# 1 = Horizontal flip (flip left to right)
# -1 = Both flips (horizontal + vertical)
    flipped = cv2.flip(img, 0)  
    cv2.imshow('original image window', img)
    cv2.imshow('flipped image window', flipped)

    cv2.imwrite('flipped.jpeg', flipped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
