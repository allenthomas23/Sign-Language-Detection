import cv2
import os
import numpy as np
from vision_processing.landmark_extraction import LandmarkExtractor
from vision_processing.landmark_drawing import LandmarkDrawer
from data_processing.folder_setup import setup_data_folders
from app_config import GESTURE_LABELS, NUM_SEQUENCES, SEQUENCE_LENGTH

class DataCollector:
    def __init__(self, base_data_path='Data'):
        self.base_data_path = base_data_path
        self.landmark_extractor = LandmarkExtractor()
        self.landmark_drawer = LandmarkDrawer()
        self.cap = cv2.VideoCapture(0)

    def collect_data(self):
        """Main loop for collecting gesture data."""
        setup_data_folders(base_dir=self.base_data_path, labels=GESTURE_LABELS, num_sequences=NUM_SEQUENCES)

        for gesture in GESTURE_LABELS:
            for seq_num in range(NUM_SEQUENCES):
                self.record_sequence(gesture, seq_num)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return

    def record_sequence(self, gesture, seq_num):
        """Records a single sequence of frames for a gesture."""
        for frame_idx in range(SEQUENCE_LENGTH):
            ret, frame = self.cap.read()
            if not ret:
                break

            image, results = self.landmark_extractor.process_frame(frame)
            self.landmark_drawer.draw_landmarks(image, results)
            
            self.display_capture_status(image, gesture, seq_num, frame_idx)
            
            keypoints = self.landmark_extractor.extract_coordinates(results)
            self.save_keypoints(keypoints, gesture, seq_num, frame_idx)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                self.cap.release()
                cv2.destroyAllWindows()
                exit()

    def display_capture_status(self, image, gesture, seq_num, frame_idx):
        """Displays status text on the recording window."""
        if frame_idx == 0:
            cv2.putText(image, 'PREPARING TO RECORD', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
            cv2.putText(image, f'Gesture: {gesture} | Video: {seq_num}', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('Data Collection', image)
            cv2.waitKey(2000)
        else:
            cv2.putText(image, f'Gesture: {gesture} | Video: {seq_num}', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('Data Collection', image)

    def save_keypoints(self, keypoints, gesture, seq_num, frame_idx):
        """Saves the extracted keypoints to a .npy file."""
        npy_path = os.path.join(self.base_data_path, gesture, str(seq_num), str(frame_idx))
        np.save(npy_path, keypoints)

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.landmark_extractor.close()

if __name__ == '__main__':
    collector = DataCollector()
    collector.collect_data()
