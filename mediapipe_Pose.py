#install MediaPipe Python package, write to cmd
#pip install mediapipe

import cv2 #opencv
import mediapipe as mp

#set up media pipe- create two variables
mp_drawing = mp.solutions.drawing_utils  #drawing utilities
mp_pose = mp.solutions.pose #importing pose estimation model (there are more models, for example face)


#if file_name = 0 it will show real-time webcam source
#otherwise you can write the path to your video
file_name = 0
if file_name == 0:
    cap = cv2.VideoCapture(file_name)
else:
    cap = cv2.VideoCapture(file_name)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('file_name_out.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

#setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Null.Frames")
            break

        # Get fps.
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Get height and width.
        h, w = image.shape[:2]


        # Recolor image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        keypoints = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = keypoints.pose_landmarks.landmark
        except:
            pass 
         
        # Render detections
        mp_drawing.draw_landmarks(image, keypoints.pose_landmarks, mp_pose.POSE_CONNECTIONS)               
        
        
        if file_name == 0:
            cv2.imshow('Mediapipe Pose', image)
        else:
            imS = cv2.resize(image, (640, 360))
            cv2.imshow('Mediapipe Pose', imS)
            out.write(image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
if file_name != 0:
    out.release()
cv2.destroyAllWindows()
