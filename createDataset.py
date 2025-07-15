import os
import numpy as np
def createDataset(data_dir='Data',actions=None, num_sequences=30, sequence_length=30):
    DATA_DIR = os.path.join(data_dir)
    actions = np.array(['hello', 'thanks', 'iloveyou']) if actions is None else actions
    #videos
    num_sequences = 30
    #frames
    sequence_length = 30
    for action in actions:
        for sequence in range(num_sequences):
            os.makedirs(os.path.join(DATA_DIR,action, str(sequence)), exist_ok=True)
if __name__ == "__main__":
    createDataset()