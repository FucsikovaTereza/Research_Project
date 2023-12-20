import cv2
import mediapipe as mp
import numpy as np
import time
import os
import csv
import datetime

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Input video file path
video_path = 0
#video_path = 'test_videos/horizontal/2022-11-14 09-26-55-1.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video details
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create video writer
if video_path == 0:
    out_video = None
    output_stats_path = f'variables_{timestamp}.csv'
else:
    # Output video path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = f'result_videos/{video_name}_3D.mp4'
    output_stats_path = f'result_statistics/{video_name}_3D.csv'
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_stats_path), exist_ok=True)
    
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    

# Open CSV file for writing statistics
with open(output_stats_path, mode='w', newline='') as csvfile:
    fieldnames = ['video_timestamp','x', 'y', 'z', 'Distance_p1_p2', 'Head orientation']
    csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    csv_writer.writeheader()

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            break

        start = time.perf_counter()
        
        # Get the timestamp
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Current time in seconds
        video_timestamp = str(datetime.timedelta(seconds=current_time))

        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False

        # Get the result
        results = face_mesh.process(image)

        # To improve performance
        image.flags.writeable = True

        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                # See where the user's head tilting
                if y < -10:
                    text = "Looking Left"
                elif y > 10:
                    text = "Looking Right"
                elif x < -10:
                    text = "Looking Down"
                elif x > 10:
                    text = "Looking Up"
                else:
                    text = "Forward"

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                cv2.line(image, p1, p2, (255, 255, 0), 3)

                # Add the text on the image
                cv2.putText(image, f'Head orientation: {text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, "x: " + str(np.round(x, 2)), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(image, "y: " + str(np.round(y, 2)), (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(image, "z: " + str(np.round(z, 2)), (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                # Save statistics to CSV
                distance_p1_p2 = np.linalg.norm(np.array(p1) - np.array(p2))
                cv2.putText(image, f'Distance: {np.round(distance_p1_p2, 2)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                csv_writer.writerow({'video_timestamp': video_timestamp, 'x': x, 'y': y, 'z': z, 'Distance_p1_p2': distance_p1_p2, 'Head orientation': text})

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

        end = time.perf_counter()
        totalTime = end - start

        fps = 1 / totalTime
        #cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        # Write frame to output video
        if out_video is not None:
            out_video.write(image)

        cv2.imshow('Head Pose Estimation', image)

        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Head Pose Estimation', cv2.WND_PROP_VISIBLE) < 1:
            break

# Release video capture and writer
cap.release()
if out_video is not None:
    out_video.release()

# Close CSV file
csvfile.close()

# Destroy all OpenCV windows
cv2.destroyAllWindows()          