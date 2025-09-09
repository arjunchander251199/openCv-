#frame by frame processing means - capture frame , process it(make rectanges , detect faces ,smiles, color ,frame speed , etc.) , REALTIME AI (SELF DRIVING CARS) , snap chat face filters , write it to a video file

#videocapture method - open camera/video file
#videoWriter method - write to video file(each frame is a image)
#realse method - video saving/writing done .Now turn off the camera 
#destroyAllWindows method - close all windows


#cv2.VideoWriter(filename , codec , fps , frame_size)
#filename - video file name
#codec - mp4,avi
#frames per second -30,60,100,120,240,300
#frame size - width , height
import cv2
#turn on web camera , 0 = First camera (default/built-in webcam)
cap = cv2.VideoCapture(0)

#frame width and height . store is as int as wee need whole number , open cv needs whole number
frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

codec=cv2.VideoWriter_fourcc(*'mp4v')
# codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

#storing write fucntion in varibale called recorded 
recorder = cv2.VideoWriter('output.mp4',codec ,30,(frame_width,frame_height))

while True:
    ret , frame = cap.read()

    #frame is found
    if ret == True:
        recorder.write(frame)
        cv2.imshow('frame window recording live',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
        
cap.release() #turn off camera 
cv2.destroyAllWindows() # free the memory 







