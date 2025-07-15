import cv2
import mediapipe as mp

class LandmarkDrawer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.drawing_specs = {
            "face": self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            "pose": self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            "hand": self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        }
        self.connection_specs = {
            "face": self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
            "pose": self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
            "hand": self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        }

    def draw_landmarks(self, image, results):
        """Draws styled landmarks on the image."""
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                self.drawing_specs["face"], self.connection_specs["face"]
            )
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                self.drawing_specs["pose"], self.connection_specs["pose"]
            )
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.drawing_specs["hand"], self.connection_specs["hand"]
            )
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.drawing_specs["hand"], self.connection_specs["hand"]
            )
