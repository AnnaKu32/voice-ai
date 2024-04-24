from speech_to_text import initialize_recognizer, list_audio_devices, capture_audio, recognize_speech
from preprocessing import preprocess_text

def main():
    recognizer = initialize_recognizer()
    list_audio_devices()
    audio = capture_audio(recognizer, device_index=1)
    text = recognize_speech(recognizer, audio)
    if text:
        print(preprocess_text(text))
    

if __name__ == "__main__":
    main()