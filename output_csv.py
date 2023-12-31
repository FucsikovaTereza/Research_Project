import cv2
import mediapipe as mp
import math
import csv
import os
import datetime

# Configuration
view = 'front'
side = 'right'
file_name = 'C:/Users/fucsi/Desktop/JADERKA/Ing/2rocnik/Research_Project/videos/2022-11-18 10-43-49-1.mkv'

# Output folder for CSV
output_folder_csv = 'result_statistics'

# Create result folder if it doesn't exist
os.makedirs(output_folder_csv, exist_ok=True)

# CSV file path
csv_file_path = os.path.join(output_folder_csv, f'{os.path.splitext(os.path.basename(file_name))[0]}_mp.csv')

# Mediapipe setup
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

# Initialize CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    if view == 'front':
        writer.writerow(["video_timestamp", "shoulders_angle", "head_right_shoulder_angle", 
                         "cervical_spine_angle", "head_pitch_yaw_angle", "eye_ear_dist",
                         "nose X", "nose Y",
                         "right_shoulder X", "right_shoulder Y",
                         "left_shoulder X", "left_shoulder Y",
                         "middle_x", "middle_y",
                         "left_eye_outer X", "left_eye_outer Y",
                         "middle_point_x", "middle_point_y",
                         "left_ear X", "left_ear Y"
                         ])
    elif view == 'side':
        if side == 'right':
            writer.writerow(["video_timestamp", "neck_inclination", "torso_inclination",
                             "right_shoulder X", "right_shoulder Y",
                             "right_hip X", "right_hip Y",
                             "right_ear X", "right_ear Y"])
        else:
            writer.writerow(["video_timestamp", "neck_inclination", "torso_inclination",
                             "left_shoulder X", "left_shoulder Y",
                             "left_hip X", "left_hip Y",
                             "left_ear X", "left_ear Y"])
        
cap = cv2.VideoCapture(file_name)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("End of video or error reading video frame.")
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
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
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
                
                # Calculate the middle point between nose and left shoulder
                middle_point_x = (nose.x + left_shoulder.x) / 2
                middle_point_y = (nose.y + left_shoulder.y) / 2
    
                # Open the CSV file in write mode
                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    # Write the data to the CSV file
                    writer.writerow([video_timestamp, shoulders_angle, head_right_shoulder_angle, 
                                     cervical_spine_angle, head_pitch_yaw_angle, eye_ear_dist,
                                     int(nose.x * w), int(nose.y * h),
                                     int(right_shoulder.x * w), int(right_shoulder.y * h),
                                     int(left_shoulder.x * w), int(left_shoulder.y * h),
                                     int(middle_x * w), int(middle_y * h),
                                     int(left_eye_outer.x * w), int(left_eye_outer.y * h),
                                     int(middle_point_x * w), int(middle_point_y * h),
                                     int(left_ear.x * w), int(left_ear.y * h)
                                     ])                                   
            
            elif view == 'side': 
                
                # Select shoulder based on side
                if side == 'right':
                    shoulder = right_shoulder
                    ear = right_ear
                    hip = right_hip
                else:  # Default to left if not right
                    shoulder = left_shoulder
                    ear = left_ear
                    hip = left_hip
                    
                # Calculate angles
                neck_inclination = calculate_angle2(int(shoulder.x * w), int(shoulder.y * h), int(ear.x * w), int(ear.y * h))
                torso_inclination = calculate_angle2(int(hip.x * w), int(hip.y * h), int(shoulder.x * w), int(shoulder.y * h))
                
                # Open the CSV file in write mode
                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    # Write the data to the CSV file
                    writer.writerow([video_timestamp, neck_inclination, torso_inclination,
                                    int(shoulder.x * w), int(shoulder.y * h),
                                    int(hip.x * w), int(hip.y * h),
                                    int(ear.x * w), int(ear.y * h)
                                    ])                
                
            else:
                print("You have just two options: 'front' or 'side'")
                break
            
        except AttributeError:
            pass   
cap.release()