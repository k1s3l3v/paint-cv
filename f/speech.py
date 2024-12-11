import speech_recognition as sr


color_map = {
    "красный": (0, 0, 255),
    "синий": (255, 0, 0),
    "зелёный": (0, 255, 0),
    "жёлтый": (0, 255, 255),
    "белый": (255, 255, 255),
    "фиолетовый": (128, 0, 128),
    "оранжевый": (255, 165, 0)
}


def detect_color():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say a color...")
        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio, language="ru-RU").lower()
            print(f"You said: {text}")

            for color, value in color_map.items():
                if color in text:
                    print(f"Detected color: {color}")
                    return value
            print("No color detected.")
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
    return None

def speech_wrapper(draw_color, lock):
    while True:
        detected_color = detect_color()
        if detected_color:
            draw_color = detected_color
