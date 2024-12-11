import cv2
import mediapipe as mp
import numpy as np


PEN_ICON_PATH = 'icons/pen.jpg'
RUBBER_ICON_PATH = 'icons/eraser.jpg'


def read_png(path):
    icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return cv2.resize(icon, (50, 50))


class PaintPlus:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    erase_color = (0, 0, 0)
    draw_thickness = 5
    erase_thickness = 100

    is_drawing = False
    is_erasing = False
    previous_position = None
    drawing_hand_index = None

    canvas = None

    pen_icon = read_png(PEN_ICON_PATH)
    rubber_icon = read_png(RUBBER_ICON_PATH)

    @classmethod
    def __detect_gesture(cls, hand_landmarks, hand_index):
        """Detect gestures based on hand landmarks."""

        index_tip = hand_landmarks.landmark[cls.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[cls.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[cls.mp_hands.HandLandmark.PINKY_TIP]
        base = hand_landmarks.landmark[cls.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

        def is_finger_up(tip, base):
            return tip.y < base.y

        # rock
        if (is_finger_up(index_tip, base)
                and is_finger_up(pinky_tip, base)
                and not is_finger_up(middle_tip, base)
        ):
            cls.is_drawing = True
            cls.is_erasing = False
            cls.drawing_hand_index = hand_index

        # peace
        elif is_finger_up(index_tip, base) and is_finger_up(middle_tip, base) and not is_finger_up(pinky_tip, base):
            cls.is_drawing = False
            cls.is_erasing = True
            cls.drawing_hand_index = hand_index

        else:
            cls.is_drawing = False
            cls.is_erasing = False
            if hand_index == 1:
                cls.previous_position = None
            cls.drawing_hand_index = None

    @classmethod
    def __process_hands(cls, frame, draw_color):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = cls.hands.process(rgb_frame)
        if results.multi_hand_landmarks:  # noqa
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):  # noqa
                cls.mp_drawing.draw_landmarks(frame, hand_landmarks, cls.mp_hands.HAND_CONNECTIONS)

                cls.__detect_gesture(hand_landmarks, hand_index)

            if cls.drawing_hand_index is not None:
                non_drawing_hand = 1 - cls.drawing_hand_index if len(results.multi_hand_landmarks) > 1 else None  # noqa
                if non_drawing_hand is not None:
                    hand_landmarks = results.multi_hand_landmarks[non_drawing_hand]  # noqa

                    index_tip = hand_landmarks.landmark[cls.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    h, w, c = frame.shape
                    index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

                    if cls.is_drawing:
                        if cls.previous_position is not None:
                            cv2.line(
                                cls.canvas, cls.previous_position, (index_x, index_y), draw_color, cls.draw_thickness
                            )
                        cls.previous_position = (index_x, index_y)
                    elif cls.is_erasing:
                        cv2.circle(cls.canvas, (index_x, index_y), cls.erase_thickness, cls.erase_color, -1)
                        cls.previous_position = None

    @classmethod
    def video_wrapper(cls, draw_color):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            if cls.canvas is None:
                cls.canvas = np.zeros_like(frame, dtype=np.uint8)

            # process hands and add it to canvas
            cls.__process_hands(frame, draw_color)

            combined = cv2.addWeighted(frame, 0.5, cls.canvas, 0.5, 0)
            cv2.rectangle(combined, (0, 0), (50, 50), draw_color, -1)


            if cls.is_drawing:
                combined[50:100, 0:50] = cls.pen_icon
            elif cls.is_erasing:
                combined[50:100, 0:50] = cls.rubber_icon

            cv2.imshow("Smart Paint", combined)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cls.hands.close()
        cv2.destroyAllWindows()