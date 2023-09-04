import os
import pandas as pd
import matplotlib.pyplot as plt

# Set the view: 'side' or 'front'
view = 'side'

# Load the CSV file into a pandas DataFrame
csv_file_path = 'variables_2023-08-29_17-12-10.csv'
data = pd.read_csv(csv_file_path)

if view == 'front':
    # Extract data from columns
    shoulders_angle = data['shoulders_angle']
    head_right_shoulder_angle = data['head_right_shoulder_angle']
    cervical_spine_angle = data['cervical_spine_angle']
    
    # Create a plot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.plot(shoulders_angle, label='Shoulders Angle', color='green')
    plt.plot(head_right_shoulder_angle, label='Head Right Shoulder Angle', color='blue')
    plt.plot(cervical_spine_angle, label='Cervical Spine Angle', color='cyan')
    
    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Angle')
    plt.title('Front Sitting Body Posture Measurements')
    
    # Display the plot
    plt.legend()
    plt.grid(True)
    
    # Save the plot as a PDF file in the same directory as the CSV file
    csv_directory, csv_filename = os.path.split(csv_file_path)
    csv_filename_without_extension = os.path.splitext(csv_filename)[0]
    pdf_file_path = os.path.join(csv_directory, f'{csv_filename_without_extension}_front_stat.pdf')
    plt.savefig(pdf_file_path, format='pdf')
    
    plt.show()

elif view == 'side':
    # Extract data from columns
    neck_inclination = data['neck_inclination']
    torso_inclination = data['torso_inclination']
    
    # Create a plot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.plot(neck_inclination, label='Neck Inclination', color='green')
    plt.plot(torso_inclination, label='Torso Inclination', color='cyan')
    
    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Angle')
    plt.title('Side Sitting Body Posture Measurements')
    
    # Display the plot
    plt.legend()
    plt.grid(True)
    
    # Save the plot as a PDF file in the same directory as the CSV file
    csv_directory, csv_filename = os.path.split(csv_file_path)
    csv_filename_without_extension = os.path.splitext(csv_filename)[0]
    pdf_file_path = os.path.join(csv_directory, f'{csv_filename_without_extension}_side_stat.pdf')
    plt.savefig(pdf_file_path, format='pdf')
    
    plt.show()
    
else:
    print("You have just two options: 'front' or 'side'")

