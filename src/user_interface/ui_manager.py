import cv2

class UIManager:
    def __init__(self):
        self.colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

    def draw_probability_bars(self, frame, probabilities, labels):
        """Draws horizontal bars representing prediction probabilities."""
        for i, (prob, label) in enumerate(zip(probabilities, labels)):
            bar_width = int(prob * 100)
            cv2.rectangle(frame, (0, 60 + i * 40), (bar_width, 90 + i * 40), self.colors[i % len(self.colors)], -1)
            cv2.putText(frame, label, (0, 85 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def display_detected_sentence(self, frame, sentence):
        """Displays the sequence of detected words at the top of the frame."""
        cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(frame, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def show_reset_message(self, frame):
        """Displays a message to the user to reset their position."""
        cv2.putText(frame, 'PAUSED: RESET POSITION', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.imshow('Sign Language Detection', frame)
        cv2.waitKey(2000)
