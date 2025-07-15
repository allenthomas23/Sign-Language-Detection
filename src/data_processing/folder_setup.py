import os
import numpy as np

def setup_data_folders(base_dir='Data', labels=None, num_sequences=30):
    """
    Creates the directory structure for storing gesture data.
    
    Args:
        base_dir (str): The root directory for the data.
        labels (list): A list of action names.
        num_sequences (int): The number of sequence folders to create for each action.
    """
    if labels is None:
        labels = []
        
    for label in labels:
        for seq_num in range(num_sequences):
            folder_path = os.path.join(base_dir, label, str(seq_num))
            os.makedirs(folder_path, exist_ok=True)
    print(f"Data folder structure created in '{base_dir}'.")

if __name__ == '__main__':
    from app_config import GESTURE_LABELS, NUM_SEQUENCES
    setup_data_folders(labels=GESTURE_LABELS, num_sequences=NUM_SEQUENCES)
