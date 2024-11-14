import os
import cv2 as cv
import csv
import numpy as np
import argparse

TRSH = 50.0

# Function to process a single video file
def process_video(video_path, output_video_path, csv_output_path):
    # Open video file
    video_capture = cv.VideoCapture(video_path)

    # Get video properties
    fps = video_capture.get(cv.CAP_PROP_FPS)
    width = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))

    # Video writer to save the output video with marks
    video_writer = cv.VideoWriter(output_video_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    prev_frame = 0

    # Prepare to save CSV data
    with open(csv_output_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Frame', 'MSE', 'Inactive'])

        frame_number = 0
        # Loop through the video frame by frame
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            
            if not ret:
                break  # End of video

            # get MSE
            curr_mse = np.mean(
                np.square(
                    frame**2 - prev_frame**2
                )
            )

            # Write the center coordinates to the CSV file
            csv_writer.writerow(
                [
                    frame_number,
                    curr_mse,
                    curr_mse  < TRSH
                ]
            )

            # Draw mse value            
            # Using cv2.putText() method
            frame = cv.putText(
                frame, 
                f'{curr_mse:04f}', 
                org=(50, 50),
                fontFace=cv.FONT_HERSHEY_SIMPLEX, 
                fontScale=1,
                color=(255, 0, 0),
                thickness=2, 
                lineType=cv.LINE_AA
            )

            # Write the frame with the red dot into the output video
            video_writer.write(frame)
            
            frame_number += 1

            pev_frame = frame

    # Release video capture and writer
    video_capture.release()
    video_writer.release()

    print(f'Processed video: {video_path}')
    print(f'CSV saved to: {csv_output_path}')
    print(f'Video saved to: {output_video_path}')

# Main function to process all videos in a folder
def main(input_folder):
    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.mp4'):
            video_path = os.path.join(input_folder, filename)

            # Define output paths based on the input file name
            output_video_path = os.path.join(input_folder, filename.replace('.mp4', '_activity.mp4'))
            csv_output_path = os.path.join(input_folder, filename.replace('.mp4', '_activity.csv'))

            # Process the video
            process_video(video_path, output_video_path, csv_output_path)

if __name__ == '__main__':
    # Argument parsing to get the folder input from the command line
    parser = argparse.ArgumentParser(description='Process videos in a folder and save predictions.')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the folder containing video files')

    args = parser.parse_args()

    # Call the main function with the input folder
    main(args.input_folder)
