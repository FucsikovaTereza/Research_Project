import cv2
import csv
import os

# Configuration
view = 'front'

# Configuration
csv_file_name = '2022-11-18 10-43-49-1_mp.csv'  # Update with your CSV file name
video_file_name = 'C:/Users/fucsi/Desktop/JADERKA/Ing/2rocnik/Research_Project/videos/2022-11-18 10-43-49-1.mkv'  # Update with your video file path
output_folder_videos = 'result_videos' 
os.makedirs(output_folder_videos, exist_ok=True)

# Paths
csv_file_path = os.path.join('result_statistics', csv_file_name)
video_output_path = os.path.join(output_folder_videos, f'{os.path.splitext(os.path.basename(video_file_name))[0]}_mp.mp4')

# Load CSV data
csv_data = []
with open(csv_file_path, mode='r', newline='') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row
    for row in reader:
        csv_data.append(row)

# Load video
cap = cv2.VideoCapture(video_file_name)
ret, frame = cap.read()
if not ret:
    print("Failed to read video file")
    cap.release()
    exit()

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(video_output_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))

# Font for text on video
font = cv2.FONT_HERSHEY_SIMPLEX

# Process video
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get data for the current frame from CSV
    if frame_idx < len(csv_data):
        frame_data = csv_data[frame_idx]
        
        if view == 'front':
            # Unpack data and assign to variables (example for 'front' view)
            video_timestamp, shoulders_angle, head_right_shoulder_angle, cervical_spine_angle, head_pitch_yaw_angle, eye_ear_dist, nose_x, nose_y, right_shoulder_x, right_shoulder_y, left_shoulder_x, left_shoulder_y, middle_x, middle_y, left_eye_outer_x, left_eye_outer_y, middle_point_x, middle_point_y, left_ear_x, left_ear_y = frame_data

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
            left_eye_outer_x = int(float(left_eye_outer_x))
            left_eye_outer_y = int(float(left_eye_outer_y))
            middle_point_x = int(float(middle_point_x))
            middle_point_y = int(float(middle_point_y))
            left_ear_x = int(float(left_ear_x))
            left_ear_y = int(float(left_ear_y))

            # Display points
            cv2.circle(frame, (nose_x, nose_y), 6, (0, 255, 0), -1)
            cv2.circle(frame, (left_shoulder_x, left_shoulder_y), 6, (255, 0, 0), -1)
            cv2.circle(frame, (middle_x, middle_y), 6, (255, 255, 0), -1)
            cv2.circle(frame, (left_eye_outer_x, left_eye_outer_y), 6, (0, 255, 255), -1)
                
            # Display angle and lines on the image
            cv2.putText(frame, f'Shoulders Angle: {shoulders_angle:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.line(frame, (nose_x, nose_y), (middle_point_x, middle_point_y), (0, 255, 0), 2)
            cv2.line(frame, (right_shoulder_x, right_shoulder_y), (nose_x, nose_y), (0, 255, 0), 2)
                
            cv2.putText(frame, f'Head-shoulder Angle: {head_right_shoulder_angle:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.line(frame, (left_shoulder_x, left_shoulder_y), (right_shoulder_x, right_shoulder_y), (255, 0, 0), 2)
            cv2.line(frame, (left_shoulder_x, left_shoulder_y), (middle_point_x, middle_point_y), (255, 0, 0), 2)
            
            cv2.putText(frame, f'Cervical Spine Angle: {cervical_spine_angle:.2f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.line(frame, (middle_x, middle_y), (middle_x, middle_y - 230), (255, 255, 0), 2)
            cv2.line(frame, (middle_x, middle_y), (right_shoulder_x, right_shoulder_y), (255, 255, 0), 2)
            
            cv2.putText(frame, f'Head pitch&yaw angle: {head_pitch_yaw_angle:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.line(frame, (left_eye_outer_x, left_eye_outer_y), (nose_x, nose_y), (0, 255, 255), 2)
            cv2.line(frame, (left_eye_outer_x, left_eye_outer_y), (left_ear_x, left_ear_y), (0, 255, 255), 2)
            
            cv2.putText(frame, f'Eye-ear Distance: {eye_ear_dist:.2f}', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)
            cv2.line(frame, (left_eye_outer_x, left_eye_outer_y), (left_ear_x, left_ear_y), (128, 0, 128), 2)
                
        elif view == 'side':
            # Unpack data and assign to variables (example for 'side' view)
            video_timestamp, neck_inclination, torso_inclination, shoulder_x, shoulder_y, hip_x, hip_y, ear_x, ear_y = frame_data

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
            cv2.putText(frame, f'Neck inclination: {neck_inclination:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.line(frame, (shoulder_x, shoulder_y), (shoulder_x, shoulder_y - 100), (0, 255, 0), 2)
            cv2.line(frame, (shoulder_x, shoulder_y), (ear_x, ear_y), (0, 255, 0), 2)
            
            cv2.putText(frame, f'Torso inclination: {torso_inclination:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.line(frame, (hip_x, hip_y), (hip_x, hip_y - 100), (255, 255, 0), 2)
            cv2.line(frame, (hip_x, hip_y), (shoulder_x, shoulder_y), (255, 255, 0), 2)

        out.write(frame)
    else:
        break  # No more data to display
    frame_idx += 1

cap.release()
out.release()