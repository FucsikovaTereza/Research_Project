import cv2
import mediapipe as mp
import math
import csv
import os
import datetime

view = 'front'
#file_name = 0
file_name = 'C:/Users/fucsi/2022-11-18 10-43-49-2_mp.csv'
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Output folders
output_folder_videos = 'result_videos' #'result_videos'
output_folder_csv = 'result_statistics'

# Create result folders if they don't exist
os.makedirs(output_folder_videos, exist_ok=True)
os.makedirs(output_folder_csv, exist_ok=True)

if file_name == 0:
    csv_file_path = f'variables_{timestamp}.csv'
else:
    csv_file_path = os.path.join(output_folder_csv, f'{os.path.splitext(os.path.basename(file_name))[0]}_mp.csv')
    video_output_path = os.path.join(output_folder_videos, f'{os.path.splitext(os.path.basename(file_name))[0]}_mp.mp4')


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_distance(a, b):
    return math.sqrt((a.y - a.x)**2 + (b.y - b.x)**2)

def calculate_angle(a, b, c):
    # Calculate the angle between three points in degrees
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    #radians = math.atan2(c[2] - b[2], c[1] - b[1]) - math.atan2(a[2] - b[2], a[1] - b[1])
    angle = math.degrees(radians)
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle


# find angle between two vectors
# first vector joining two points (x1, y1) and (x2, y2)
# second vector joining point (x1, y1) and any point on the y-axis passing through (x1, y1), in our case we set it at (x1, 0)
def calculate_angle2(x1, y1, x2, y2):
    theta = math.acos( (y2 -y1)*(-y1) / (math.sqrt(
        (x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
    angle = int(180/math.pi)*theta
    return angle

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    if view == 'front':
        writer.writerow(["video_timestamp", "shoulders_angle", "head_right_shoulder_angle", 
                         "cervical_spine_angle", "head_pitch_yaw_angle", "eye_ear_dist"])
    elif view == 'side':
        writer.writerow(["neck_inclination", "torso_inclination"])
                
if file_name == 0:
    cap = cv2.VideoCapture(file_name)
else:
    cap = cv2.VideoCapture(file_name)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(video_output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Null.Frames")
            break

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            h, w = image.shape[:2]
    
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
    
            keypoints = pose.process(image)
    
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
            landmarks = keypoints.pose_landmarks.landmark
            
            # Get coordinates of specific landmarks
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            left_eye_outer = landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER]
            
            # Get the timestamp
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Current time in seconds
            video_timestamp = str(datetime.timedelta(seconds=current_time))
            
            if view == 'front':
                
                # Calculate angles
                shoulders_angle = calculate_angle(left_shoulder, nose, right_shoulder)
                head_right_shoulder_angle = calculate_angle(nose, left_shoulder, right_shoulder)
                head_pitch_yaw_angle = calculate_angle(nose, left_eye_outer, left_ear)
                
                middle_x = (right_shoulder.x + left_shoulder.x) / 2
                middle_y = (right_shoulder.y + left_shoulder.y) / 2
                cervical_spine_angle = calculate_angle2(int(middle_x * w), int(middle_y * h), int(left_shoulder.x * w), int(left_shoulder.y * h))
                
                eye_ear_dist = calculate_distance(left_ear, left_eye_outer)
    
                # Open the CSV file in write mode
                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    # Write the data to the CSV file
                    writer.writerow([video_timestamp, shoulders_angle, head_right_shoulder_angle, 
                                     cervical_spine_angle, head_pitch_yaw_angle, eye_ear_dist])            
                        
                # Display points
                cv2.circle(image, (int(nose.x * w), int(nose.y * h)), 6, (0, 255, 0), -1)
                cv2.circle(image, (int(left_shoulder.x * w), int(left_shoulder.y * h)), 6, (255, 0, 0), -1)
                cv2.circle(image, (int(middle_x * w), int(middle_y * h)), 6, (255, 255, 0), -1)
                cv2.circle(image, (int(left_eye_outer.x * w), int(left_eye_outer.y * h)), 6, (0, 255, 255), -1)
                
                # Calculate the middle point between nose and left shoulder
                middle_point_x = (nose.x + left_shoulder.x) / 2
                middle_point_y = (nose.y + left_shoulder.y) / 2
                
                # Display angle and lines on the image
                cv2.putText(image, f'Shoulders Angle: {shoulders_angle:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.line(image, (int(nose.x * w), int(nose.y * h)), (int(middle_point_x * w), int(middle_point_y * h)), (0, 255, 0), 2)
                cv2.line(image, (int(right_shoulder.x * w), int(right_shoulder.y * h)), (int(nose.x * w), int(nose.y * h)), (0, 255, 0), 2)
                
                cv2.putText(image, f'Head-shoulder Angle: {head_right_shoulder_angle:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.line(image, (int(left_shoulder.x * w), int(left_shoulder.y * h)), (int(right_shoulder.x * w), int(right_shoulder.y * h)), (255, 0, 0), 2)
                cv2.line(image, (int(left_shoulder.x * w), int(left_shoulder.y * h)), (int(middle_point_x * w), int(middle_point_y * h)), (255, 0, 0), 2)
                
                cv2.putText(image, f'Cervical Spine Angle: {cervical_spine_angle:.2f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.line(image, (int(middle_x * w), int(middle_y * h)), (int(middle_x * w), int(middle_y * h) - 230), (255, 255, 0), 2)
                cv2.line(image, (int(middle_x * w), int(middle_y * h)), (int(right_shoulder.x * w), int(right_shoulder.y * h)), (255, 255, 0), 2)
                
                cv2.putText(image, f'Head pitch&yaw angle: {head_pitch_yaw_angle:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.line(image, (int(left_eye_outer.x * w), int(left_eye_outer.y * h)), (int(nose.x * w), int(nose.y * h)), (0, 255, 255), 2)
                cv2.line(image, (int(left_eye_outer.x * w), int(left_eye_outer.y * h)), (int(left_ear.x * w), int(left_ear.y * h)), (0, 255, 255), 2)
                
                cv2.putText(image, f'Eye-ear Distance: {eye_ear_dist:.2f}', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)
                cv2.line(image, (int(left_eye_outer.x * w), int(left_eye_outer.y * h)), (int(left_ear.x * w), int(left_ear.y * h)), (128, 0, 128), 2)
            
            elif view == 'side': 
                
                # Calculate angles
                neck_inclination = calculate_angle2(int(left_shoulder.x * w), int(left_shoulder.y * h), int(left_ear.x * w), int(left_ear.y * h))
                torso_inclination = calculate_angle2(int(left_hip.x * w), int(left_hip.y * h), int(left_shoulder.x * w), int(left_shoulder.y * h))
                
                # Open the CSV file in write mode
                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    # Write the data to the CSV file
                    writer.writerow([video_timestamp, neck_inclination, torso_inclination])
                
                # Display points
                cv2.circle(image, (int(left_shoulder.x * w), int(left_shoulder.y * h)), 6, (0, 255, 0), -1)
                cv2.circle(image, (int(left_hip.x * w), int(left_hip.y * h)), 6, (255, 255, 0), -1)
                
                # Display angle and lines on the image
                cv2.putText(image, f'Neck inclination: {neck_inclination:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.line(image, (int(left_shoulder.x * w), int(left_shoulder.y * h)), (int(left_shoulder.x * w), int(left_shoulder.y * h) - 100), (0, 255, 0), 2)
                cv2.line(image, (int(left_shoulder.x * w), int(left_shoulder.y * h)), (int(left_ear.x * w), int(left_ear.y * h)), (0, 255, 0), 2)
                
                cv2.putText(image, f'Torso inclination: {torso_inclination:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.line(image, (int(left_hip.x * w), int(left_hip.y * h)), (int(left_hip.x * w), int(left_hip.y * h) - 100), (255, 255, 0), 2)
                cv2.line(image, (int(left_hip.x * w), int(left_hip.y * h)), (int(left_shoulder.x * w), int(left_shoulder.y * h)), (255, 255, 0), 2)
                
            else:
                print("You have just two options: 'front' or 'side'")
                break
            
        except AttributeError:
            pass
        

        if file_name == 0:
            cv2.imshow('Mediapipe Pose', image)
        #else:
            #imS = cv2.resize(image, (640, 360))
            #cv2.imshow('Mediapipe Pose', imS)
            #out.write(image)

        #if cv2.waitKey(5) & 0xFF == ord('q') or cv2.getWindowProperty('Mediapipe Pose', cv2.WND_PROP_VISIBLE) < 1:
            #break

cap.release()
if file_name != 0:
    out.release()
#cv2.destroyAllWindows()