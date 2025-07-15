import numpy as np
GESTURE_LABELS = np.array(['hello', 'thanks', 'deaf'])

#number of sequences (videos) for each action
NUM_SEQUENCES = 30
#number of frames in each sequence
SEQUENCE_LENGTH = 30
MODEL_PATH = 'action.h5'
#Confidence threshold for displaying a prediction
DETECTION_THRESHOLD = 0.8
#Number of recent predictions to display on screen
SENTENCE_HISTORY_LENGTH = 5
