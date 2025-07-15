import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from app_config import GESTURE_LABELS, NUM_SEQUENCES, SEQUENCE_LENGTH

class DatasetBuilder:
    def __init__(self, base_data_path='Data'):
        self.base_data_path = base_data_path
        self.gesture_map = {label: num for num, label in enumerate(GESTURE_LABELS)}

    def load_data(self):
        """
        Loads the saved .npy files and organizes them into sequences and labels.
        """
        sequences, labels = [], []
        for gesture, label_idx in self.gesture_map.items():
            for seq_num in range(NUM_SEQUENCES):
                window = []
                for frame_idx in range(SEQUENCE_LENGTH):
                    res = np.load(os.path.join(self.base_data_path, gesture, str(seq_num), f"{frame_idx}.npy"))
                    window.append(res)
                sequences.append(window)
                labels.append(label_idx)
        return np.array(sequences), np.array(labels)

    def split_dataset(self, sequences, labels, test_size=0.05):
        """
        Splits the dataset into training and testing sets.
        """
        x = np.array(sequences)
        y = to_categorical(labels).astype(int)
        return train_test_split(x, y, test_size=test_size)

if __name__ == '__main__':
    builder = DatasetBuilder()
    sequences, labels = builder.load_data()
    x_train, x_test, y_train, y_test = builder.split_dataset(sequences, labels)
    print("Training data shape:", x_train.shape)
    print("Testing data shape:", x_test.shape)
