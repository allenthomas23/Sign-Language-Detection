import cv2
import mediapipe as mp
import numpy as np

class LandmarkExtractor:
    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        """
        Processes a single video frame to detect landmarks.
        
        Args:
            frame: The input video frame from OpenCV.
            
        Returns:
            A tuple containing the processed image and the landmark results.
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return annotated_image, results

    def extract_coordinates(self, results):
        """
        Extracts and flattens the coordinates of all detected landmarks.
        
        Args:
            results: The landmark results from MediaPipe.
            
        Returns:
            A single NumPy array containing all landmark data.
        """
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([pose, face, lh, rh])

    def close(self):
        self.holistic.close()
