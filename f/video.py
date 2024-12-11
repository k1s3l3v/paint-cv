import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe and OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

erase_color = (0, 0, 0)  # Black for eraser (acts like erasing on a black background)
draw_thickness = 5
erase_thickness = 100

# State variables
is_drawing = False
is_erasing = False
previous_position = None  # To track previous position for continuous drawing
drawing_hand_index = None

# Initialize a blank canvas
canvas = None

def detect_gesture(hand_landmarks, hand_index):
    """Detect gestures based on hand landmarks."""
    global is_drawing, is_erasing, drawing_hand_index, previous_position

    # Get landmark positions
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    # Helper function to calculate if a finger is up
    def is_finger_up(tip, base):
        return tip.y < base.y

    if is_finger_up(index_tip, base) and is_finger_up(pinky_tip, base) and not is_finger_up(thumb_tip, base) and not is_finger_up(middle_tip, base):
        print("drawing enabled")
        is_drawing = True
        is_erasing = False
        drawing_hand_index = hand_index

    # Check if "peace" gesture (index and middle finger extended)
    elif is_finger_up(index_tip, base) and is_finger_up(middle_tip, base) and not is_finger_up(pinky_tip, base):
        print("erasing enabled")
        is_drawing = False
        is_erasing = True
        drawing_hand_index = hand_index

    else:
        is_drawing = False
        is_erasing = False
        print("waiting")
        if hand_index == 1:
            previous_position = None
        drawing_hand_index = None


def video_wrapper(draw_color):
    global canvas, drawing_hand_index, previous_position, is_drawing, is_erasing

    # Start video capture
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for natural interaction
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Initialize the canvas if it's the first frame
        if canvas is None:
            canvas = np.zeros_like(frame, dtype=np.uint8)

        # Process the frame for hand landmarks
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks: # noqa
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks): # noqa
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Detect the gesture
                detect_gesture(hand_landmarks, hand_index)

            # Check if we are drawing or erasing with the non-gesture hand
            if drawing_hand_index is not None:
                non_drawing_hand = 1 - drawing_hand_index if len(results.multi_hand_landmarks) > 1 else None # noqa
                if non_drawing_hand is not None:
                    hand_landmarks = results.multi_hand_landmarks[non_drawing_hand] # noqa

                    # Get the position of the index finger tip
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    h, w, c = frame.shape
                    index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

                    # Draw or erase on the canvas based on the gesture
                    if is_drawing:
                        if previous_position is not None:
                            cv2.line(canvas, previous_position, (index_x, index_y), (0, 255, 0), draw_thickness)
                        previous_position = (index_x, index_y)
                    elif is_erasing:
                        cv2.circle(canvas, (index_x, index_y), erase_thickness, erase_color, -1)
                        previous_position = None

        # Combine the frame and the canvas
        combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        # Show the result
        cv2.imshow("Smart Paint", combined)

        # Break the loop with the ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release resources
    cap.release()
    hands.close()
    cv2.destroyAllWindows()