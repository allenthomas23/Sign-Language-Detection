import cv2
import numpy as np
import mediapipe as mp

# drawing utils
mp_drawing = mp.solutions.drawing_utils
# bring in holistic model
mp_holistic = mp.solutions.holistic

"""
This function runs the MediaPipe model on the image
and returns the image with the results which has all the landmarks
"""
def mediapipe_detection(image, model):
    # MediaPipe expects RGB images
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    # Convert back to BGR for OpenCV display
    annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return annotated, results

# function takes in the latest frame and returns the image with the landmarks
def draw_landmarks(image, results):
    # draw the left hand
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )
    # draw the right hand
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )
    # body pose
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS
    )
    # face mesh
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION
    )

# only add if the frame detects the respective hand
# otherwise return zeros
def extract_landmarks(results):
    # Right hand: 21 points × 4 values each
    if results.right_hand_landmarks:
        right = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility]
             for lm in results.right_hand_landmarks.landmark]
        ).flatten()
    else:
        # 21 points × 4 values each  zeros
        right = np.zeros(21 * 4)

    # Left hand: 21 points × 3 values each (no visibility)
    if results.left_hand_landmarks:
        left = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.left_hand_landmarks.landmark]
        ).flatten()
    else:
        left = np.zeros(21 * 3)
    if results.pose_landmarks:
        # Pose: 33 points × 3 values each
        pose = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.pose_landmarks.landmark]
        ).flatten()
    else:
        # 33 points × 3 values each  zeros
        pose = np.zeros(33 * 3)
    if results.face_landmarks:
        # Face: 468 points × 3 values each
        face = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.face_landmarks.landmark]
        ).flatten()
    else:
        # 468 points × 3 values each  zeros
        face = np.zeros(468 * 3)
    return np.concatenate([pose,face,left,right])


def main():
    
    cap = cv2.VideoCapture(0)
    # initialize holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # run detection
            image, results = mediapipe_detection(frame, holistic)
            # draw landmarks on the frame
            draw_landmarks(image, results)

            # extract landmarks for further processing
            allLandmarks = extract_landmarks(results)

            cv2.imshow('MediaPipe Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(allLandmarks)

if __name__ == '__main__':
    main()
