import cv2
import mediapipe as mp
import math
import csv
import datetime


class MediaPipe_PoseEstimation_csv_output: 
    # Initializes the MediaPipe_PoseEstimation class
    def __init__(self, input_file, csv_file_name, view, side):
        self.input_file = input_file
        self.csv_file_name = csv_file_name
        self.view = view
        self.side = side
        
    def calculate_distance(self,a, b):
        return math.sqrt((a.y - a.x)**2 + (b.y - b.x)**2)
        
    # Calculates the angle between three points using the arctangent method
    def calculate_angle(self, a, b, c):
        radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
        angle = math.degrees(radians)
        angle = abs(angle)
        if angle > 180:
            angle = 360 - angle
        return angle

    # Calculates the angle between two points and one of the axes
    def calculate_angle2(self, x1, y1, x2, y2, axis='y', orientation='right'):
        if (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * x1) != 0:
            if axis == 'x':
                theta = math.acos((x2 - x1) * (-x1) / (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * x1))
            elif axis == 'y':
                theta = math.acos((y2 - y1) * (-y1) / (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
            else:
                raise ValueError("Invalid axis, use 'x' or 'y'")

            if orientation == 'right':
                angle = int(180 / math.pi) * theta
            elif orientation == 'left':
                angle = 180 - int(180 / math.pi) * theta
            else:
                raise ValueError("Invalid orientation, use 'left' or 'right'")
        else:
            return 0

        return angle

    # Calculates the midpoint between two points
    def middle_point(self, a, b):
        midpoint_x = (a.x + b.x) / 2
        midpoint_y = (a.y + b.y) / 2
        return midpoint_x, midpoint_y

    # Processes the input video, calculates pose statistics, and generates an output video and write the statistics into .csv file
    def process_video(self):
        mp_pose = mp.solutions.pose

        with open(self.csv_file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            if self.view == 'front':
                writer.writerow(["video_timestamp", "shoulders_angle", "shoulders_inclination", "head_right_shoulder_angle", 
                                 "cervical_spine_angle", "head_pitch_yaw_angle", "eye_ear_dist", "head_pitch_yaw_angle2", "eye_ear_dist2",
                                 "nose X", "nose Y", "nose Z",
                                 "right_shoulder X", "right_shoulder Y", "right_shoulder Z",
                                 "left_shoulder X", "left_shoulder Y", "left_shoulder Z",
                                 "middle_x", "middle_y",
                                 "middle_point_x", "middle_point_y",
                                 "middle_point2_x", "middle_point2_y",
                                 "right_eye_outer X", "right_eye_outer Y", "right_eye_outer Z",
                                 "left_eye_outer X", "left_eye_outer Y", "left_eye_outer Z",
                                 "right_ear X", "right_ear Y", "right_ear Z",
                                 "left_ear X", "left_ear Y", "left_ear Z",
                                 "nose_world X", "nose_world Y", "nose_world Z",
                                 "right_shoulder_world X", "right_shoulder_world Y", "right_shoulder_world Z",
                                 "left_shoulder_world X", "left_shoulder_world Y", "left_shoulder_world Z",
                                 "middle_x_world", "middle_y_world",
                                 "right_eye_outer X_world", "right_eye_outer Y_world", "right_eye_outer Z_world",
                                 "left_eye_outer_world X", "left_eye_outer_world Y", "left_eye_outer_world Z",
                                 "right_ear_world X", "right_ear_world Y", "right_ear_world Z",
                                 "left_ear_world X", "left_ear_world Y", "left_ear_world Z"
                                 ])
            elif self.view == 'side':
                if self.side == 'right':
                    writer.writerow(["video_timestamp", "neck_inclination", "torso_inclination",
                                     "right_shoulder X", "right_shoulder Y", "right_shoulder Z",
                                     "right_hip X", "right_hip Y", "right_hip Z",
                                     "right_ear X", "right_ear Y", "right_ear Z",
                                     "right_shoulder_world X", "right_shoulder_world Y", "right_shoulder_world Z",
                                     "right_hip_world X", "right_hip_world Y", "right_hip_world Z",
                                     "right_ear_world X", "right_ear_world Y", "right_ear_world Z"
                                     ])
                else:
                    writer.writerow(["video_timestamp", "neck_inclination", "torso_inclination",
                                     "left_shoulder X", "left_shoulder Y", "left_shoulder Z",
                                     "left_hip X", "left_hip Y", "left_hip Z",
                                     "left_ear X", "left_ear Y", "left_ear Z",
                                     "left_shoulder_world X", "left_shoulder_world Y", "left_shoulder_world Z",
                                     "left_hip_world X", "left_hip_world Y", "left_hip_world Z",
                                     "left_ear_world X", "left_ear_world Y", "left_ear_world Z"
                                     ])

        cap = cv2.VideoCapture(self.input_file)

        frame_number = 0
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                  print("Null.Frames")
                  break
                try:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    video_timestamp = round(frame_number / fps)
                    video_timestamp = str(datetime.timedelta(seconds=video_timestamp))
                    h, w = image.shape[:2]

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False

                    keypoints = pose.process(image)

                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    landmarks = keypoints.pose_landmarks.landmark
                    world_landmarks = keypoints.pose_world_landmarks.landmark

                    # Extract Landmarks
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    nose = landmarks[mp_pose.PoseLandmark.NOSE]
                    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
                    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
                    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                    left_eye_outer = landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER]
                    right_eye_outer = landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER]
                    
                    # Extract WorldLandmarks
                    left_shoulder_world = world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder_world = world_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    nose_world = world_landmarks[mp_pose.PoseLandmark.NOSE]
                    left_ear_world = world_landmarks[mp_pose.PoseLandmark.LEFT_EAR]
                    right_ear_world = world_landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
                    left_hip_world = world_landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                    right_hip_world = world_landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                    left_eye_outer_world = world_landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER]
                    right_eye_outer_world = world_landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER]
                    
                    
                    if self.view == 'front':
                        
                        # Calculate angles
                        shoulders_angle = self.calculate_angle(left_shoulder, nose, right_shoulder)
                        head_right_shoulder_angle = self.calculate_angle(nose, left_shoulder, right_shoulder)
                        head_pitch_yaw_angle = self.calculate_angle(nose, left_eye_outer, left_ear)
                        head_pitch_yaw_angle2 = self.calculate_angle(nose, right_eye_outer, right_ear)

                        shoulders_inclination = self.calculate_angle2(int(right_shoulder.x * w), int(right_shoulder.y * h),
                                                                      int(left_shoulder.x * w), int(left_shoulder.y * h),
                                                                      'x', 'left')
                        
                        middle_x = (right_shoulder.x + left_shoulder.x) / 2
                        middle_y = (right_shoulder.y + left_shoulder.y) / 2
                        middle_x_world = (right_shoulder_world.x + left_shoulder_world.x) / 2
                        middle_y_world = (right_shoulder_world.y + left_shoulder_world.y) / 2
                        cervical_spine_angle = self.calculate_angle2(int(middle_x * w), int(middle_y * h), int(left_shoulder.x * w), int(left_shoulder.y * h))
                        
                        eye_ear_dist = self.calculate_distance(left_ear, left_eye_outer)
                        eye_ear_dist2 = self.calculate_distance(right_ear, right_eye_outer)
                        
                        # Calculate the middle point between nose and left shoulder
                        middle_point_x = (nose.x + left_shoulder.x) / 2
                        middle_point_y = (nose.y + left_shoulder.y) / 2
                        # Calculate the middle point between nose and left shoulder
                        middle_point2_x = (right_shoulder.x + middle_x) / 2
                        middle_point2_y = (right_shoulder.y + middle_y) / 2

                        # Open the CSV file in write mode
                        with open(self.csv_file_name, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            # Write the data to the CSV file
                            writer.writerow([video_timestamp, shoulders_angle, shoulders_inclination, head_right_shoulder_angle, 
                                             cervical_spine_angle, head_pitch_yaw_angle, eye_ear_dist, head_pitch_yaw_angle2, eye_ear_dist2,
                                             int(nose.x * w), int(nose.y * h), int(nose.z * w),
                                             int(right_shoulder.x * w), int(right_shoulder.y * h), int(right_shoulder.z * w),
                                             int(left_shoulder.x * w), int(left_shoulder.y * h), int(left_shoulder.z * w),
                                             int(middle_x * w), int(middle_y * h),
                                             int(middle_point_x * w), int(middle_point_y * h),
                                             int(middle_point2_x * w), int(middle_point2_y * h),
                                             int(right_eye_outer.x * w), int(right_eye_outer.y * h), int(right_eye_outer.z * w),
                                             int(left_eye_outer.x * w), int(left_eye_outer.y * h), int(left_eye_outer.z * w),
                                             int(right_ear.x * w), int(right_ear.y * h), int(right_ear.z * w),
                                             int(left_ear.x * w), int(left_ear.y * h), int(left_ear.z * w),
                                             int(nose_world.x), int(nose_world.y), int(nose_world.z),
                                             int(right_shoulder_world.x), int(right_shoulder_world.y), int(right_shoulder_world.z),
                                             int(left_shoulder_world.x), int(left_shoulder_world.y), int(left_shoulder_world.z),
                                             int(middle_x_world), int(middle_y_world),
                                             int(right_eye_outer_world.x), int(right_eye_outer_world.y), int(right_eye_outer_world.z),
                                             int(left_eye_outer_world.x), int(left_eye_outer_world.y), int(left_eye_outer_world.z),
                                             int(right_ear_world.x), int(right_ear_world.y), int(right_ear_world.z),
                                             int(left_ear_world.x), int(left_ear_world.y), int(left_ear_world.z)
                                             ])                                   
                    
                    elif self.view == 'side': 
                        
                        # Select shoulder based on side
                        if self.side == 'right':
                            shoulder = right_shoulder
                            ear = right_ear
                            hip = right_hip
                            shoulder_world = right_shoulder_world
                            ear_world = right_ear_world
                            hip_world = right_hip_world
                        else:  # Default to left if not right
                            shoulder = left_shoulder
                            ear = left_ear
                            hip = left_hip
                            shoulder_world = left_shoulder_world
                            ear_world = left_ear_world
                            hip_world = left_hip_world
                            
                        # Calculate angles
                        neck_inclination = self.calculate_angle2(int(shoulder.x * w), int(shoulder.y * h), int(ear.x * w), int(ear.y * h))
                        torso_inclination = self.calculate_angle2(int(hip.x * w), int(hip.y * h), int(shoulder.x * w), int(shoulder.y * h))
                        
                        # Open the CSV file in write mode
                        with open(self.csv_file_name, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            # Write the data to the CSV file
                            writer.writerow([video_timestamp, neck_inclination, torso_inclination,
                                            int(shoulder.x * w), int(shoulder.y * h), int(shoulder.z * w),
                                            int(hip.x * w), int(hip.y * h), int(hip.z * w),
                                            int(ear.x * w), int(ear.y * h), int(ear.z * w),
                                            int(shoulder_world.x), int(shoulder_world.y), int(shoulder_world.z),
                                            int(hip_world.x), int(hip_world.y), int(hip_world.z),
                                            int(ear_world.x), int(ear_world.y), int(ear_world.z)
                                            ])                       

                    frame_number += 1
                    
                except AttributeError:
                    pass 

        # Release the video capture and writer objects
        cap.release()
        
# Set input type
import os
input_type = 'file'
side = 'left'
view = 'front'

if input_type == 'file':
    # SET AN INPUT FILE in the folder cropped_videos
    input_file = "C:/Users/fucsi/Desktop/JADERKA/Ing/2. ročník/Research_Project/videos/2022-11-21 17-30-24-1.mkv"
    csv_file_name = f'{os.path.splitext(input_file)[0]}.csv'

    # Create 'statistics' folder if it doesn't exist
    statistics_folder = 'statistics/Honza'
    os.makedirs(statistics_folder, exist_ok=True)

    # Set the paths for csv file and output video in the 'statistics' folder
    csv_file_path = os.path.join(statistics_folder, os.path.basename(csv_file_name))

    video_processor = MediaPipe_PoseEstimation_csv_output(input_file, csv_file_path, view, side)
    video_processor.process_video()