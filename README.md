# Sign Language Detection

This project is a real-time sign language detection application that uses your webcam to recognize and translate sign language gestures into text.

## Features

-   **Real-time Gesture Recognition**: Translates sign language gestures captured from a webcam in real-time.
-   **Deep Learning Model**: Utilizes a trained LSTM model to accurately identify gestures.
-   **User-Friendly Interface**: Displays the live camera feed, detected landmarks, and translated text.

## Project Structure

```
Sign-Language-Detection/
├── Data/                     # Stores gesture data
├── logs/                     # Training logs
├── src/                      # Source code
│   ├── data_processing/      # Scripts for data preparation
│   ├── model_training/       # Model training script
│   ├── vision_processing/    # Hand landmark extraction and drawing
│   ├── user_interface/       # UI components
│   ├── main_app.py           # Main application
│   └── app_config.py         # Application configuration
├── action.h5                 # Trained model file
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Sign-Language-Detection.git
    cd Sign-Language-Detection
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate 
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Training the Model

Before running the application, you need to train the model on your own sign language gestures.

### 1. Collect Gesture Data

This script will capture video sequences for each gesture and save them as `.npy` files.

1.  **Configure Gestures**: Open `src/app_config.py` and modify the `GESTURE_LABELS` list to include the gestures you want to train.

2.  **Run the data collection script:**
    ```bash
    python src/collect_gesture_data.py
    ```
    - The script will create folders for each gesture in the `Data/` directory.
    - For each gesture, it will record a number of video sequences (`NUM_SEQUENCES`).
    - Follow the on-screen instructions to start and stop recording for each sequence.

### 2. Train the Model

Once the data is collected, you can train the LSTM model.

1.  **Run the training script:**
    ```bash
    python src/model_training/train_model.py
    ```
    - This script will preprocess the collected data, build the LSTM model, and train it.
    - The trained model will be saved as `action.h5` in the root directory.

## How to Run

1.  **Run the application:**
    ```bash
    python src/main_app.py
    ```

2.  **Perform sign language gestures** in front of your webcam. The application will display the recognized gestures on the screen.

3.  **Press 'q' to quit** the application.