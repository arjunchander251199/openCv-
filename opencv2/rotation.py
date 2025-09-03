import cv2


img = cv2.imread('image1.jpeg')
if img is None:
    print('image not found')
else:
    print("image loaded successfully")

    center=(img.shape[1]//2,img.shape[0]//2)
    #forumla to calcute how each pixel should move when rotated(recipe)
    Matrix=cv2.getRotationMatrix2D(center,90,1)#center, rotation angle, scale(more scale more zoomed)
  

  #90 - anticlockwise
  #-90 -clockwise
  #1 - scale (no zoom)
  #-1 -scale (no zoom but flipped vertically and horizontally)



    #.shape (h,w,c)
    #wrapAffine(w,h) (runs the formaula on each pixel)(cooking)
    #warpAffine(img,Matrix,dsize) (img,matrix,dsize)
    rotated=cv2.warpAffine(img,Matrix,(img.shape[1],img.shape[0]))



    cv2.imshow('original image window', img)
    cv2.imshow('rotated image window', rotated)

    cv2.imwrite('rotated.jpeg', rotated)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()



