#In image processing, a video is a sequence of images (frames) displayed rapidly one after another, which gives the perception of continuous motion.
#on an average its 30fps to 60fps

#1 frame one image , mulptiple frame makes a video

# webcam - real time video

#frame 1 , detect face , draw rectangle
#frame 2 , detect face , draw rectangle
#frame 3 , detect face , draw rectangle
#frame 4 , detect face , draw rectangle

import cv2 
#video capture means open camera/video file
# directly put the video file path , avi - audio video interleaved , mp4 - media player classic 4 /compression levels/os support
#0 = First camera (default/built-in webcam)
#1 = Second camera (external USB camera, second built-in camera, etc.)
#2 = Third camera device
#3 = Fourth camera device

cap = cv2.VideoCapture(0)

while(True):

    #cap.read() is a method that captures one frame at a time from your video source (camera or video file).
    #ret - true if a frame was successfully read, false if not
    #frame - the image frame is returned as a numpy array

    # color_img = np.array([
    # [[  0,   0, 255], [  0, 255,   0], [255,   0,   0]],  # Row 1
    # [[255, 255,   0], [  0, 255, 255], [255,   0, 255]],  # Row 2
    # [[128, 128, 128], [ 64,  64,  64], [192, 192, 192]]   # Row 3
    
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        cv2.imshow('Frame window',frame)

        #0 means frame will be updated in the window only when press key
        #1 means every 1 millisecond frame will be updated in window
        #2 means every 2 millisecond frame will be updated in window
        #3 means every 3 millisecond frame will be updated in window
        #4 means every 4 millisecond frame will be updated in window            

        # Pressing 'q' will exit the program 
        #if no key is pressed return -1 and runs the next frame
        #ord - means ordinal gives ascii value of a key q=113 , Q=81 ,a=97 , computer only understands ascii values .
        #example 0x00000113 - 0xFF makes it 113 (mac , windows ) -may return a large interger
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the capture
cap.release() #turn off my camera/ close the video
cv2.destroyAllWindows() # free the memory




# 1. cap.release()

# Frees the video source (camera or video file).

# Example: webcam is released so other apps can use it.

# Doesn’t close windows.

# 2. cv2.destroyAllWindows()

# Closes all OpenCV windows that you opened with cv2.imshow().

# Frees the memory used for displaying those images/frames.

# Doesn’t touch the camera/video file.
