import cv2
import numpy as np
import mediapipe as mp
import os

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils
# Bring in holistic model
mp_holistic = mp.solutions.holistic

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Data') 

# Videos are going to be 30 frames in length
sequence_length = 30

def mediapipe_detection(image, model):
    """
    This function runs the MediaPipe model on the image
    and returns the image with the results which has all the landmarks.
    """
    # MediaPipe expects RGB images
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    # Convert back to BGR for OpenCV display
    annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return annotated_image, results

def draw_styled_landmarks(image, results):
    """
    This function draws the styled landmarks on the image.
    """
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_landmarks(results):
    """
    Extracts the keypoints from the results object and concatenates them.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def main():
    """
    Main function to collect keypoint data for a single action.
    """
    # Get the action name from the user
    action = input("Enter the name of the action to record: ").lower()
    
    # Get the number of sequences to record
    while True:
        try:
            no_sequences = int(input(f"Enter the number of videos to record for '{action}': "))
            if no_sequences > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    
    action_path = os.path.join(DATA_PATH, action)
    start_sequence = 0
    if os.path.exists(action_path):
        # List existing sequence folders and find the max number
        existing_sequences = [int(f) for f in os.listdir(action_path) if f.isdigit()]
        if existing_sequences:
            start_sequence = max(existing_sequences) + 1
    
    print(f"\nStarting data collection for action '{action}'.")
    print(f"Recording {no_sequences} new videos, starting from video number {start_sequence}.")

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # Loop through sequences (videos)
        for sequence in range(start_sequence, start_sequence + no_sequences):
            # Create the folder for the sequence if it doesn't exist
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except FileExistsError:
                pass # Folder already exists, which is fine

            # Loop through video length (sequence length)
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Action: {action} - Video: {sequence}', (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000) # Wait 2 seconds
                else: 
                    cv2.putText(image, f'Action: {action} - Video: {sequence}', (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # Export keypoints
                keypoints = extract_landmarks(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                    
    cap.release()
    cv2.destroyAllWindows()
    print("\nData collection complete.")

if __name__ == "__main__":
    main()
