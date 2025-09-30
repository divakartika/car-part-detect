import cv2
import os
import pandas as pd
import argparse

def label_video_frames(video_path, output_dir='car_dataset_real', frame_skip=5):
    """
    Extracts frames from a video and allows the user to label them for multi-label classification.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the labeled images and CSV file.
        frame_skip (int): The number of frames to skip between labeling prompts.
    """
    images_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        
    labels_csv_path = os.path.join(output_dir, 'labels.csv')

    # --- Load existing data or create new ---
    if os.path.exists(labels_csv_path):
        df = pd.read_csv(labels_csv_path)
        records = df.to_dict('records')
        # Find the last image index to avoid overwriting
        if not df.empty:
            last_filename = df['filename'].iloc[-1]
            start_index = int(os.path.splitext(os.path.basename(last_filename))[0]) + 1
        else:
            start_index = 0
    else:
        records = []
        start_index = 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0
    img_index = start_index

    # --- Part states initialization ---
    parts = ['hood', 'front_left_door', 'front_right_door', 'rear_left_door', 'rear_right_door']
    part_states = {part: 0 for part in parts} # 0 for closed, 1 for open

    print("\n--- Video Labeling Tool ---")
    print("Press the keys 1-5 to toggle the state of a car part.")
    print("  [1] Hood: Closed")
    print("  [2] Front Left Door: Closed")
    print("  [3] Front Right Door: Closed")
    print("  [4] Rear Left Door: Closed")
    print("  [5] Rear Right Door: Closed")
    print("\nPress [s] to SAVE the current frame with its labels.")
    print("Press [n] to SKIP to the next frame without saving.")
    print("Press [q] to QUIT and save all labels collected.")
    print("---------------------------------")


    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        
        # --- Display current labels on the frame ---
        display_frame = frame.copy()
        y_offset = 30
        for i, part in enumerate(parts):
            state_text = "open" if part_states[part] == 1 else "closed"
            color = (0, 255, 0) if part_states[part] == 0 else (0, 0, 255)
            cv2.putText(display_frame, f"[{i+1}] {part}: {state_text}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30
        
        cv2.imshow('Labeling Tool - Press (s) to save, (n) to skip, (q) to quit', display_frame)
        
        # --- Handle User Input ---
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'): # Save frame and labels
            filename = f"{img_index}.png"
            relative_path = os.path.join('images', filename)
            cv2.imwrite(os.path.join(images_dir, filename), frame)
            
            label_record = {'filename': relative_path}
            label_record.update(part_states)
            records.append(label_record)
            
            print(f"Saved {filename} with labels: {part_states}")
            img_index += 1
        elif key == ord('n'): # Skip frame
            continue
        elif ord('1') <= key <= ord('5'):
            part_index = key - ord('1')
            part_to_toggle = parts[part_index]
            part_states[part_to_toggle] = 1 - part_states[part_to_toggle] # Toggle between 0 and 1
    
    # --- Cleanup and Save ---
    cap.release()
    cv2.destroyAllWindows()

    if records:
        final_df = pd.DataFrame(records)
        final_df = final_df[['filename'] + parts] # Ensure column order
        final_df.to_csv(labels_csv_path, index=False)
        print(f"\nLabeling complete. All labels saved to {labels_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Label frames from a video for car part detection.")
    parser.add_argument("video_file", type=str, help="Path to the video file to be labeled.")
    parser.add_argument("--output_dir", type=str, default="car_dataset_real", help="Directory to save the images and labels.csv file.")
    parser.add_argument("--skip", type=int, default=5, help="Number of frames to skip between each labeling action.")
    
    args = parser.parse_args()
    
    label_video_frames(args.video_file, args.output_dir, args.skip)
