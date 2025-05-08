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

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        cv2.imshow('frame', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()