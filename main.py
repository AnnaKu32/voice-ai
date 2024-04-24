from speech_to_text import initialize_recognizer, list_audio_devices, capture_audio, recognize_speech
from preprocessing import Preprocessor


def main():
    recognizer = initialize_recognizer()
    list_audio_devices()
    audio = capture_audio(recognizer)
    speech_text = recognize_speech(recognizer, audio)
    if speech_text is None:
        print("No speech was recognized. Please try again.")
        return
    print(speech_text)
    
    preprocessor = Preprocessor()
    preproces_text = preprocessor.preprocess(speech_text)
    print(preproces_text)

if __name__ == '__main__':
    main()

