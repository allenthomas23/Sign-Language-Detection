import os
import numpy as np
def main():
    DATA_DIR = os.path.join('Data')
    actions = np.array(['hello', 'thanks', 'iloveyou'])
    #videos
    num_sequences = 30
    #frames
    sequence_length = 30
    for action in actions:
        for sequence in range(num_sequences):
            os.makedirs(os.path.join(DATA_DIR,action, str(sequence)), exist_ok=True)
if __name__ == "__main__":
    main()