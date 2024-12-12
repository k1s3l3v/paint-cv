import threading

from speech import SpeechRecognizer
from video import PaintPlus


if __name__ == '__main__':
    color = [0, 255, 0]

    audio_thread = threading.Thread(target=SpeechRecognizer.speech_wrapper, args=(color, None))
    audio_thread.daemon = True
    audio_thread.start()

    PaintPlus.video_wrapper(color)
