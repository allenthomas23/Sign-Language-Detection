import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

#drawing utils
mp_drawing = mp.solutions.drawing_utils
#bring in holistic model
mp_holistic = mp.solutions.holistic

""""This function runs the media pipe model on the image
and returns the image with the results"""
def mediapipe_detection(image, model):
    #mediapipe needs rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image) 
    image.flags.writeable = True 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results
def draw_landmarks(image, results):
    #draw the left hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                               mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))
    #draw the right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                               mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)
        cv2.imshow('frame', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
print(results.right_hand_landmarks)
cv2.destroyAllWindows()