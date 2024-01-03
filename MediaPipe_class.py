import cv2
import mediapipe as mp
import math
import csv
import datetime

class MediaPipe_PoseEstimation:
    # Initializes the MediaPipe_PoseEstimation class
    def __init__(self, input_file, csv_file_name, output_video_name, view, side):
        self.input_file = input_file
        self.csv_file_name = csv_file_name
        self.output_video_name = output_video_name
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
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(self.output_video_name, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

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

                    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))              

                    landmarks = results.pose_landmarks.landmark
                    world_landmarks = results.pose_world_landmarks.landmark
                    
                    # Plot pose world landmarks.
                    if frame_number == 1:
                        mp_drawing = mp.solutions.drawing_utils 
                        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
                        #cv2.imwrite('statistics/Honza2/frame-2.jpg', image)

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

                        shoulders_inclination = self.calculate_angle2(right_shoulder.x, right_shoulder.y,
                                                                      left_shoulder.x, left_shoulder.y,
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
                                             round(nose.x * w, 2), round(nose.y * h, 2), round(nose.z * w, 2),
                                             round(right_shoulder.x * w, 2), round(right_shoulder.y * h, 2), round(right_shoulder.z * w, 2),
                                             round(left_shoulder.x * w, 2), round(left_shoulder.y * h, 2), round(left_shoulder.z * w, 2),
                                             round(middle_x * w, 2), round(middle_y * h, 2),
                                             round(middle_point_x * w, 2), round(middle_point_y * h, 2),
                                             round(middle_point2_x * w, 2), round(middle_point2_y * h, 2),
                                             round(right_eye_outer.x * w, 2), round(right_eye_outer.y * h, 2), round(right_eye_outer.z * w, 2),
                                             round(left_eye_outer.x * w, 2), round(left_eye_outer.y * h, 2), round(left_eye_outer.z * w, 2),
                                             round(right_ear.x * w, 2), round(right_ear.y * h, 2), round(right_ear.z * w, 2),
                                             round(left_ear.x * w, 2), round(left_ear.y * h, 2), round(left_ear.z * w, 2),
                                             round(nose_world.x, 2), round(nose_world.y, 2), round(nose_world.z, 2),
                                             round(right_shoulder_world.x, 2), round(right_shoulder_world.y, 2), round(right_shoulder_world.z, 2),
                                             round(left_shoulder_world.x, 2), round(left_shoulder_world.y, 2), round(left_shoulder_world.z, 2),
                                             round(middle_x_world, 2), round(middle_y_world, 2),
                                             round(right_eye_outer_world.x, 2), round(right_eye_outer_world.y, 2), round(right_eye_outer_world.z, 2),
                                             round(left_eye_outer_world.x, 2), round(left_eye_outer_world.y, 2), round(left_eye_outer_world.z, 2),
                                             round(right_ear_world.x, 2), round(right_ear_world.y, 2), round(right_ear_world.z, 2),
                                             round(left_ear_world.x, 2), round(left_ear_world.y, 2), round(left_ear_world.z, 2)
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
                        neck_inclination = self.calculate_angle2(shoulder.x, shoulder.y, ear.x, ear.y)
                        torso_inclination = self.calculate_angle2(hip.x, hip.y, shoulder.x, shoulder.y)
                        
                        # Open the CSV file in write mode
                        with open(self.csv_file_name, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            # Write the data to the CSV file
                            writer.writerow([video_timestamp, neck_inclination, torso_inclination,
                                            round(shoulder.x * w, 2), round(shoulder.y * h, 2), round(shoulder.z * w, 2),
                                            round(hip.x * w, 2), round(hip.y * h, 2), round(hip.z * w, 2),
                                            round(ear.x * w, 2), round(ear.y * h, 2), round(ear.z * w, 2),
                                            round(shoulder_world.x, 2), round(shoulder_world.y, 2), round(shoulder_world.z, 2),
                                            round(hip_world.x, 2), round(hip_world.y, 2), round(hip_world.z, 2),
                                            round(ear_world.x, 2), round(ear_world.y, 2), round(ear_world.z, 2)
                                            ])                       

                    # Display points
                    if self.view == 'front':       
                        cv2.circle(image, (int(nose.x * w), int(nose.y * h)), 6, (0, 255, 0), -1)
                        cv2.circle(image, (int(left_shoulder.x * w), int(left_shoulder.y * h)), 6, (255, 0, 0), -1)
                        cv2.circle(image, (int(middle_x * w), int(middle_y * h)), 6, (255, 255, 0), -1)
                        cv2.circle(image, (int(left_eye_outer.x * w), int(left_eye_outer.y * h)), 6, (0, 255, 255), -1)
                        cv2.circle(image, (int(right_shoulder.x * w), int(right_shoulder.y * h)), 6, (0, 150, 255), -1)
                            
                        # Display angle and lines on the image
                        cv2.putText(image, f'Shoulders Angle: {shoulders_angle:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.line(image, (int(nose.x * w), int(nose.y * h)), (int(middle_point_x * w), int(middle_point_y * h)), (0, 255, 0), 2)
                        cv2.line(image, (int(right_shoulder.x * w), int(right_shoulder.y * h)), (int(nose.x * w), int(nose.y * h)), (0, 255, 0), 2)
                        
                        cv2.putText(image, f'Head-shoulder Angle: {head_right_shoulder_angle:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        cv2.line(image, (int(left_shoulder.x * w), int(left_shoulder.y * h)), (int(right_shoulder.x * w), int(right_shoulder.y * h)), (255, 0, 0), 2)
                        cv2.line(image, (int(left_shoulder.x * w), int(left_shoulder.y * h)), (int(middle_point_x * w), int(middle_point_y * h)), (255, 0, 0), 2)
                        
                        cv2.putText(image, f'Cervical Spine Angle: {cervical_spine_angle:.2f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        cv2.line(image, (int(middle_x * w), int(middle_y * h)), (int(middle_x * w), int(middle_y * h) - 230), (255, 255, 0), 2)
                        cv2.line(image, (int(middle_x * w), int(middle_y * h)), (int(middle_point2_x * w), int(middle_point2_y * h)), (255, 255, 0), 2)
                        
                        cv2.putText(image, f'Head pitch&yaw angle: {head_pitch_yaw_angle:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.line(image, (int(left_eye_outer.x * w), int(left_eye_outer.y * h)), (int(nose.x * w), int(nose.y * h)), (0, 255, 255), 2)
                        cv2.line(image, (int(left_eye_outer.x * w), int(left_eye_outer.y * h)), (int(left_ear.x * w), int(left_ear.y * h)), (0, 255, 255), 2)
                        
                        cv2.putText(image, f'Eye-ear Distance: {eye_ear_dist:.2f}', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)
                        cv2.line(image, (int(left_eye_outer.x * w), int(left_eye_outer.y * h)), (int(left_ear.x * w), int(left_ear.y * h)), (128, 0, 128), 2)
                        
                        cv2.putText(image, f'Shoulders inclination: {shoulders_inclination:.2f}', (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 255), 2)
                        cv2.line(image, (int(right_shoulder.x * w), int(right_shoulder.y * h)), (int(right_shoulder.x * w) + 230, int(right_shoulder.y * h)), (0, 150, 255), 2)
                        cv2.line(image, (int(middle_point2_x * w), int(middle_point2_y * h)), (int(right_shoulder.x * w), int(right_shoulder.y * h)), (0, 150, 255), 2)
                            
                    elif self.view == 'side':   
                        cv2.circle(image, (int(shoulder.x * w), int(shoulder.y * h)), 6, (0, 255, 0), -1)
                        cv2.circle(image, (int(hip.x * w), int(hip.y * h)), 6, (255, 255, 0), -1)
                        
                        # Display angle and lines on the image
                        cv2.putText(image, f'Neck inclination: {neck_inclination:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.line(image, (int(shoulder.x * w), int(shoulder.y * h)), (int(shoulder.x * w), int(shoulder.y * h) - 230), (0, 255, 0), 2)
                        cv2.line(image, (int(shoulder.x * w), int(shoulder.y * h)), (int(ear.x * w), int(ear.y * h)), (0, 255, 0), 2)
                        
                        cv2.putText(image, f'Torso inclination: {torso_inclination:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        cv2.line(image, (int(hip.x * w), int(hip.y * h)), (int(hip.x * w), int(hip.y * h) - 230), (255, 255, 0), 2)
                        cv2.line(image, (int(hip.x * w), int(hip.y * h)), (int(shoulder.x * w), int(shoulder.y * h)), (255, 255, 0), 2)

                    # Write the frame into the file
                    out.write(image)

                    #if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit the video window
                    #    break

                    frame_number += 1

                except Exception as e:
                    print(f"An error occurred: {e}")

        # Release the video capture and writer objects
        cap.release()
        out.release()

        # Destroy all OpenCV windows
        #cv2.destroyAllWindows()