import cv2
import csv

class MediaPipe_PoseEstimation_video_output:
    # Initializes the MediaPipe_PoseEstimation class
    def __init__(self, video_file_name, csv_file_path, video_output_path, view, side):
        self.video_file_name = video_file_name
        self.csv_file_path = csv_file_path
        self.video_output_path = video_output_path
        self.view = view
        self.side = side

    # Processes the input video, calculates pose statistics, and generates an output video and write the statistics into .csv file
    def process_video(self):        
        # Load CSV data
        csv_data = []
        with open(self.csv_file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                csv_data.append(row)
        
        # Load video
        cap = cv2.VideoCapture(self.video_file_name)
        ret, frame = cap.read()
        if not ret:
            print("Failed to read video file")
            cap.release()
            exit()
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(self.video_output_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))

        # Process video
        frame_idx = 0
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
        
            # Get data for the current frame from CSV
            if frame_idx < len(csv_data)-23: # there is 23 world coordinates
                frame_data = csv_data[frame_idx]
                
                if view == 'front':
                    # Unpack data and assign to variables (example for 'front' view)
                    video_timestamp, shoulders_angle, shoulders_inclination, head_right_shoulder_angle, cervical_spine_angle, head_pitch_yaw_angle, eye_ear_dist, head_pitch_yaw_angle2, eye_ear_dist2, nose_x, nose_y, nose_z, right_shoulder_x, right_shoulder_y, right_shoulder_z, left_shoulder_x, left_shoulder_y, left_shoulder_z, middle_x, middle_y, middle_point_x, middle_point_y, middle_point2_x, middle_point2_y, right_eye_outer_x, right_eye_outer_y, right_eye_outer_z, left_eye_outer_x, left_eye_outer_y, left_eye_outer_z, right_ear_x, right_ear_y, right_ear_z, left_ear_x, left_ear_y, left_ear_z = frame_data
        
                    # Convert numerical values to float and pixel values to integer
                    shoulders_angle = float(shoulders_angle)
                    head_right_shoulder_angle = float(head_right_shoulder_angle)
                    cervical_spine_angle = float(cervical_spine_angle)
                    head_pitch_yaw_angle = float(head_pitch_yaw_angle)
                    eye_ear_dist = float(eye_ear_dist)
        
                    nose_x = int(float(nose_x))
                    nose_y = int(float(nose_y))
                    right_shoulder_x = int(float(right_shoulder_x))
                    right_shoulder_y = int(float(right_shoulder_y))
                    left_shoulder_x = int(float(left_shoulder_x))
                    left_shoulder_y = int(float(left_shoulder_y))
                    middle_x = int(float(middle_x))
                    middle_y = int(float(middle_y))
                    middle_point_x = int(float(middle_point_x))
                    middle_point_y = int(float(middle_point_y))
                    middle_point2_x = int(float(middle_point2_x))
                    middle_point2_y = int(float(middle_point2_y))
                    right_eye_outer_x = int(float(right_eye_outer_x))
                    right_eye_outer_y = int(float( right_eye_outer_y))
                    left_eye_outer_x = int(float(left_eye_outer_x))
                    left_eye_outer_y = int(float(left_eye_outer_y))
                    right_ear_x = int(float(right_ear_x))
                    right_ear_y = int(float(right_ear_y))
                    left_ear_x = int(float(left_ear_x))
                    left_ear_y = int(float(left_ear_y))

                    # Display points
                    if self.view == 'front':       
                        cv2.circle(image, (nose_x, nose_y), 6, (0, 255, 0), -1)
                        cv2.circle(image, (left_shoulder_x, left_shoulder_y), 6, (255, 0, 0), -1)
                        cv2.circle(image, (middle_x, middle_y), 6, (255, 255, 0), -1)
                        cv2.circle(image, (left_eye_outer_x, left_eye_outer_y), 6, (0, 255, 255), -1)
                        cv2.circle(image, (right_shoulder_x, right_shoulder_y), 6, (0, 150, 255), -1)
                            
                        # Display angle and lines on the image
                        cv2.putText(image, f'Shoulders Angle: {shoulders_angle:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.line(image, (nose_x, nose_y), (middle_point_x, middle_point_y), (0, 255, 0), 2)
                        cv2.line(image, (right_shoulder_x, right_shoulder_y), (nose_x, nose_y), (0, 255, 0), 2)
                        
                        cv2.putText(image, f'Head-shoulder Angle: {head_right_shoulder_angle:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        cv2.line(image, (left_shoulder_x, left_shoulder_y), (right_shoulder_x, right_shoulder_y), (255, 0, 0), 2)
                        cv2.line(image, (left_shoulder_x, left_shoulder_y), (middle_point_x, middle_point_y), (255, 0, 0), 2)
                        
                        cv2.putText(image, f'Cervical Spine Angle: {cervical_spine_angle:.2f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        cv2.line(image, (middle_x, middle_y), (middle_x, middle_y - 230), (255, 255, 0), 2)
                        cv2.line(image, (middle_x, middle_y), (middle_point2_x, middle_point2_y), (255, 255, 0), 2)
                        
                        cv2.putText(image, f'Head pitch&yaw angle: {head_pitch_yaw_angle:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.line(image, (left_eye_outer_x, left_eye_outer_y), (nose_x, nose_y), (0, 255, 255), 2)
                        cv2.line(image, (left_eye_outer_x, left_eye_outer_y), (left_ear_x, left_ear_y), (0, 255, 255), 2)
                        
                        cv2.putText(image, f'Eye-ear Distance: {eye_ear_dist:.2f}', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)
                        cv2.line(image, (left_eye_outer_x, left_eye_outer_y), (left_ear_x, left_ear_y), (128, 0, 128), 2)
                        
                        cv2.putText(image, f'Shoulders inclination: {shoulders_inclination:.2f}', (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 255), 2)
                        cv2.line(image, (right_shoulder_x, right_shoulder_y), (right_shoulder_x + 230, right_shoulder_y), (0, 150, 255), 2)
                        cv2.line(image, (middle_point2_x, middle_point2_y), (right_shoulder_x, right_shoulder_y), (0, 150, 255), 2)

                    elif view == 'side':
                        # Unpack data and assign to variables (example for 'side' view)
                        video_timestamp, neck_inclination, torso_inclination, shoulder_x, shoulder_y, shoulder_z, hip_x, hip_y, hip_z, ear_x, ear_y, ear_z = frame_data
            
                        # Convert string data to float for numerical values and to integer for pixel values
                        neck_inclination = float(neck_inclination)
                        torso_inclination = float(torso_inclination)
                        shoulder_x = int(float(shoulder_x))
                        shoulder_y = int(float(shoulder_y))
                        hip_x = int(float(hip_x))
                        hip_y = int(float(hip_y))
                        ear_x = int(float(ear_x))
                        ear_y = int(float(ear_y))
            
                        # Display points
                        cv2.circle(frame, (shoulder_x, shoulder_y), 6, (0, 255, 0), -1)
                        cv2.circle(frame, (hip_x, hip_y), 6, (255, 255, 0), -1)
                        
                        # Display angle and lines on the image
                        cv2.putText(image, f'Neck inclination: {neck_inclination:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.line(image, (shoulder_x, shoulder_y), (shoulder_x, shoulder_y - 100), (0, 255, 0), 2)
                        cv2.line(image, (shoulder_x, shoulder_y), (ear_x, ear_y), (0, 255, 0), 2)
                        
                        cv2.putText(image, f'Torso inclination: {torso_inclination:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        cv2.line(image, (hip_x, hip_y), (hip_x, hip_y - 100), (255, 255, 0), 2)
                        cv2.line(image, (hip_x, hip_y), (shoulder_x, shoulder_y), (255, 255, 0), 2)
            
                    out.write(image)
                else:
                    break  # No more data to display
                frame_idx += 1
            
            cap.release()
            out.release()


# Set input type
import os
input_type = 'file'
side = 'left'
view = 'front'

if input_type == 'file':
    csv_file_name = 'Honza/2022-11-21 17-30-24-1.csv'
    video_file_name = 'C:/Users/fucsi/Desktop/JADERKA/Ing/2. ročník/Research_Project/videos/2022-11-21 17-30-24-1.mkv'

    # Create 'statistics' folder if it doesn't exist
    output_folder_videos = 'statistics/Honza' 
    os.makedirs(output_folder_videos, exist_ok=True)

    # Set the paths for csv file and output video in the 'statistics' folder
    csv_file_path = os.path.join('statistics', csv_file_name)
    video_output_path = os.path.join(output_folder_videos, f'{os.path.splitext(os.path.basename(video_file_name))[0]}_mp.mp4')

    video_processor = MediaPipe_PoseEstimation_video_output(video_file_name, csv_file_path, video_output_path, view, side)
    video_processor.process_video()