import numpy as np

# Define the actions that the model will recognize
GESTURE_LABELS = np.array(['hello', 'thanks', 'deaf'])

# --- Data Collection ---
# Number of sequences (videos) for each action
NUM_SEQUENCES = 30
# Number of frames in each sequence
SEQUENCE_LENGTH = 30

# --- Model ---
# Path to save the trained model
MODEL_PATH = 'action.h5'

# --- Real-time Detection ---
# Confidence threshold for displaying a prediction
DETECTION_THRESHOLD = 0.8
# Number of recent predictions to display on screen
SENTENCE_HISTORY_LENGTH = 5
