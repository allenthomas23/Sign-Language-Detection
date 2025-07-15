from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
import numpy as np
num_sequences = 30
actions = ['hello', 'thanks', 'deaf']
frames = 30
def createLabelMap():
    
    labels = {label:num for num, label in enumerate(actions)}
    return labels

def appendArrays():
    """
    Process the data from the numpy files and create sequences and labels.
    essentially appends all the numpy arrays for one video then has a matching label array for that action
    """
    sequences, labels = [], []
    for action in actions:
        #for each video
        for sequence in range(num_sequences):
            window = []
            for frame_num in range(frames):
                result = np.load(os.path.join('data',action, str(sequence),"{}.npy".format(frame_num)))
                window.append(result)
            sequences.append(window)
            labels.append(label_map[action])
    return sequences, labels

def datasetSplit(sequences,labels):
    x = np.array(sequences)
    y = to_categorical(labels).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
if __name__ == '__main__':
    label_map = createLabelMap()
    print(label_map)
    sequences, labels = appendArrays()
    datasetSplit(sequences, labels)
