from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from app_config import GESTURE_LABELS, MODEL_PATH

class GestureModel:
    def __init__(self, input_shape=(30, 1662)):
        self.input_shape = input_shape
        self.num_classes = len(GESTURE_LABELS)
        self.model = self._build_model()

    def _build_model(self):
        """Builds and compiles the LSTM model."""
        model = Sequential([
            LSTM(64, return_sequences=True, activation='relu', input_shape=self.input_shape),
            LSTM(128, return_sequences=True, activation='relu'),
            LSTM(64, return_sequences=False, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model

    def train(self, x_train, y_train, epochs=100):
        """Trains the model."""
        self.model.fit(x_train, y_train, epochs=epochs)
        self.save_model()

    def save_model(self):
        """Saves the trained model to a file."""
        self.model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

if __name__ == '__main__':
    from data_processing.prepare_dataset import DatasetBuilder
    

    builder = DatasetBuilder()
    sequences, labels = builder.load_data()
    x_train, x_test, y_train, y_test = builder.split_dataset(sequences, labels)
    

    gesture_recognizer = GestureModel(input_shape=(x_train.shape[1], x_train.shape[2]))
    gesture_recognizer.train(x_train, y_train)
    

    loss, accuracy = gesture_recognizer.model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
