import cv2

# --------------------------------------------
# Haar Cascades - Face, Eye, Smile Detection
# - Pretrained ML model (Decision Trees)
# - Detects objects using rectangles only
# - Fast and works in real-time
# - Open source → freely available in OpenCV
# - Works offline (needs XML files only)
# Pretrained XML files: 
# https://github.com/opencv/opencv/tree/master/data/haarcascades
# --------------------------------------------

# Load Haar Cascade models
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a single frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Loop through each detected face (x, y, w, h)
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        # cv2.rectangle(image, top-left, bottom-right, color, thickness)
        # → top-left = (x, y), bottom-right = (x+w, y+h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop face region (ROI)
        # Formula: ROI_gray = gray[y : y+h, x : x+w], ROI_color = frame[y : y+h, x : x+w]
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes inside face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=25)
        for (x, y, w, h) in eyes:
            # Draw rectangle around each eye
            cv2.rectangle(roi_color, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Detect smile inside face ROI
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=25)
        for (x, y, w, h) in smiles:
            # Draw rectangle around each smile
            cv2.rectangle(roi_color, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Put text above face when smile is detected
            # cv2.putText(image, text, org, font, fontScale, color, thickness)
            # → image      : the image/frame
            # → text       : string to display
            # → org        : bottom-left corner of text (x, y)
            # → font       : font type (cv2.FONT_HERSHEY_SIMPLEX)
            # → fontScale  : size of text (1.0 = normal, 0.5 = smaller)
            # → color      : BGR color (0,255,255 = yellow)
            # → thickness  : thickness of text lines
            cv2.putText(frame, "Smile :)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)

    # Show final frame
    cv2.imshow('Face, Eye & Smile Detection', frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
