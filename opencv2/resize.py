import cv2

img = cv2.imread('image1.jpeg')

if img is None:
    print('image not found')
else:
    print("image loaded successfully")
    resized = cv2.resize(img, (500, 500)) #width, height
    cv2.imshow('original image window', img)
    cv2.imshow('resized image window', resized)

    cv2.imwrite('resized.jpeg', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


