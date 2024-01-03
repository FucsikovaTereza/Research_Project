import os
import pandas as pd
import matplotlib.pyplot as plt

view = 'side'

# Load the CSV file into a pandas DataFrame
csv_file_path = "statistics/Honza/2022-11-21 17-30-24-2.csv"
data = pd.read_csv(csv_file_path)


# Convert 'video_timestamp' to datetime and calculate minutes since start
video_timestamps = pd.to_datetime(data['video_timestamp'], format='%H:%M:%S')
start_time = video_timestamps.iloc[0]
minutes_since_start = video_timestamps.apply(lambda x: (x - start_time).total_seconds() / 60)

# Selecting every 150th row from the dataset
sampling_interval = 90  # Adjust this value as needed
data_sampled = data.iloc[::sampling_interval]
minutes_since_start_sampled = minutes_since_start.iloc[::sampling_interval]

if view == 'front':
    
    # Creating three subplots under each other with a wider figure width
    fig, axs = plt.subplots(6, 1, figsize=(15, 18))
    
    # Add a title to the figure
    fig.suptitle('Body Posture Measurements Over Time- front view', fontsize=16, verticalalignment='top', y=0.98)
    
    # Define custom colors
    darker_yellow = (220/255, 220/255, 15/255)
    wine_color = (128/255, 0/255, 128/255)
    
    # Shoulders Angle Plot
    axs[0].plot(minutes_since_start_sampled, data_sampled['shoulders_angle'], color='green', label='Shoulders Angle')
    axs[0].set_xlabel('Minutes')
    axs[0].set_ylabel('Degrees')
    axs[0].grid(True)
    axs[0].legend(fontsize='large')
    axs[0].set_xlim(left=0, right=32)
    
    # Head Right Shoulder Angle Plot
    axs[1].plot(minutes_since_start_sampled, data_sampled['head_right_shoulder_angle'], color='blue', label='Head Right Shoulder Angle')
    axs[1].set_xlabel('Minutes')
    axs[1].set_ylabel('Degrees')
    axs[1].grid(True)
    axs[1].legend(fontsize='large')
    axs[1].set_xlim(left=0, right=32)
    
    # Cervical Spine Angle Plot
    axs[2].plot(minutes_since_start_sampled, data_sampled['cervical_spine_angle'], color='cyan', label='Cervical Spine Angle')
    axs[2].set_xlabel('Minutes')
    axs[2].set_ylabel('Degrees')
    axs[2].grid(True)
    axs[2].legend(fontsize='large')
    axs[2].set_xlim(left=0, right=32)
    
    # Head Pitch Yaw Angle Plot
    axs[3].plot(minutes_since_start_sampled, data_sampled['head_pitch_yaw_angle'], color= darker_yellow, label='Head Pitch Yaw Angle')
    axs[3].set_xlabel('Minutes')
    axs[3].set_ylabel('Degrees')
    axs[3].grid(True)
    axs[3].legend(fontsize='large')
    axs[3].set_xlim(left=0, right=32)
    
    # Eye Ear Distance Plot
    axs[4].plot(minutes_since_start_sampled, data_sampled['eye_ear_dist'], color=wine_color, label='Eye Ear Distance')  # Burgundy approximated as darkred
    axs[4].set_xlabel('Minutes')
    axs[4].set_ylabel('Distance (pixels)')
    axs[4].grid(True)
    axs[4].legend(fontsize='large')
    axs[4].set_xlim(left=0, right=32)

    # Shoulders Inclination Plot
    axs[5].plot(minutes_since_start_sampled, data_sampled['shoulders_inclination'], color='orange', label='Shoulders Inclination')
    axs[5].set_xlabel('Minutes')
    axs[5].set_ylabel('Degrees')
    axs[5].grid(True)
    axs[5].legend(fontsize='large')
    axs[5].set_xlim(left=0, right=32)
    
    # Display the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) 
    
    # Save the plot as a PDF file in the same directory as the CSV file
    csv_directory, csv_filename = os.path.split(csv_file_path)
    csv_filename_without_extension = os.path.splitext(csv_filename)[0]
    pdf_file_path = os.path.join(csv_directory, f'{csv_filename_without_extension}_front_stat.pdf')
    plt.savefig(pdf_file_path, format='pdf')
    
    plt.show()

elif view == 'side':
    
    # Creating three subplots under each other with a wider figure width
    fig, axs = plt.subplots(2, 1, figsize=(15, 8))
    
    # Add a title to the figure
    fig.suptitle('Body Posture Measurements Over Time- side view', fontsize=16, verticalalignment='top', y=0.98)
    
    # Shoulders Angle Plot
    axs[0].plot(minutes_since_start_sampled, data_sampled['neck_inclination'], color='green', label='Neck Inclination')
    axs[0].set_xlabel('Minutes')
    axs[0].set_ylabel('Degrees')
    axs[0].grid(True)
    axs[0].legend(fontsize='large')
    axs[0].set_xlim(left=0, right=32)
    
    # Head Right Shoulder Angle Plot
    axs[1].plot(minutes_since_start_sampled, data_sampled['torso_inclination'], color='cyan', label='Torso Inclination')
    axs[1].set_xlabel('Minutes')
    axs[1].set_ylabel('Degrees')
    axs[1].grid(True)
    axs[1].legend(fontsize='large')
    axs[1].set_xlim(left=0, right=32)
    
    
    # Display the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) 
    
    # Save the plot as a PDF file in the same directory as the CSV file
    csv_directory, csv_filename = os.path.split(csv_file_path)
    csv_filename_without_extension = os.path.splitext(csv_filename)[0]
    pdf_file_path = os.path.join(csv_directory, f'{csv_filename_without_extension}_side_stat.pdf')
    plt.savefig(pdf_file_path, format='pdf')
    
    plt.show()

    
else:
    print("You have just two options: 'front' or 'side'")

