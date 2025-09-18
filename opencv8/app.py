# Haar Cascades
# - Pretrained machine learning model (uses Decision Trees)
# - Used for object detection (faces, eyes, cars, etc.)-only rectangles
# - Very fast compared to other ML methods
# - Works in real-time (good for webcams and live video)
# - Open source → freely available in OpenCV
# - Works offline (no internet needed once you have the XML file)
#
# Pretrained XML files are here:
# https://github.com/opencv/opencv/tree/master/data/haarcascades

import cv2

# Load the pre-trained Haar Cascade face detection model
# (OpenCV already provides this XML file)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a single frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale 
    # (Haar cascades work on black & white images for faster processing)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------------
    # detectMultiScale() → Detect objects in the image
    #
    # Parameters:
    #   gray          → the input image (must be grayscale)
    #   scaleFactor   → how much the image is reduced at each scale
    #                   Example: 1.1 = shrink by 10% each step
    #                   Smaller values = more accuracy, slower speed
    #                   Larger values = faster, but may miss objects
    #
    #   minNeighbors  → how many nearby detections are required to 
    #                   confirm an object - 3 overapping rectangles in a frame
    #                   Example: 3 = loose (may detect false faces)
    #                            5 = safe (balanced, common choice)
    #                            7+ = strict (only strong detections)
    #
    #   
    # --------------------------------------------------------
    detect_face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Loop through all detected faces and draw rectangles around them
#That rectangle is defined by 4 numbers:

# x → X-coordinate of top-left corner

# y → Y-coordinate of top-left corner

# w → width of rectangle

# h → height of rectangle
    for (x, y, w, h) in detect_face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green box

    # Show the frame with detections
    cv2.imshow('Face Detection', frame)

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
