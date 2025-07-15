import cv2
import numpy as np
from tensorflow.keras.models import load_model
from vision_processing.landmark_extraction import LandmarkExtractor
from vision_processing.landmark_drawing import LandmarkDrawer
from user_interface.ui_manager import UIManager
from app_config import GESTURE_LABELS, SEQUENCE_LENGTH, DETECTION_THRESHOLD, SENTENCE_HISTORY_LENGTH, MODEL_PATH

class RealTimeDetector:
    def __init__(self):
        self.model = load_model(MODEL_PATH)
        self.landmark_extractor = LandmarkExtractor()
        self.landmark_drawer = LandmarkDrawer()
        self.ui_manager = UIManager()
        self.sequence_data = []
        self.detected_gestures = []
        self.cap = cv2.VideoCapture(0)

    def run(self):
        """Main loop for real-time gesture detection."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            image, results = self.landmark_extractor.process_frame(frame)
            self.landmark_drawer.draw_landmarks(image, results)
            
            keypoints = self.landmark_extractor.extract_coordinates(results)
            self.sequence_data.append(keypoints)
            self.sequence_data = self.sequence_data[-SEQUENCE_LENGTH:]

            if len(self.sequence_data) == SEQUENCE_LENGTH:
                self.predict_and_display(image)

            self.ui_manager.display_detected_sentence(image, self.detected_gestures)
            cv2.imshow('Sign Language Detection', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        self.cleanup()

    def predict_and_display(self, image):
        """Predicts the gesture and updates the UI."""
        prediction = self.model.predict(np.expand_dims(self.sequence_data, axis=0))[0]
        
        if prediction[np.argmax(prediction)] > DETECTION_THRESHOLD:
            predicted_gesture = GESTURE_LABELS[np.argmax(prediction)]
            
            if not self.detected_gestures or predicted_gesture != self.detected_gestures[-1]:
                self.detected_gestures.append(predicted_gesture)
                self.detected_gestures = self.detected_gestures[-SENTENCE_HISTORY_LENGTH:]

        self.ui_manager.draw_probability_bars(image, prediction, GESTURE_LABELS)
        self.ui_manager.show_reset_message(image)
        self.sequence_data = [] # Reset sequence after prediction

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.landmark_extractor.close()

if __name__ == '__main__':
    detector = RealTimeDetector()
    detector.run()
