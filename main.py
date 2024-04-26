from speech_to_text import initialize_recognizer, list_audio_devices, capture_audio, recognize_speech
from preprocessing import pred_class, get_response
from model import classes, words, data
from convert_text_to_speech import convert_text_to_speech

def main():
    recognizer = initialize_recognizer()
    list_audio_devices()
   
    while True:
        audio = capture_audio(recognizer)
        speech_text = recognize_speech(recognizer, audio)
        if speech_text is None:
            print("No speech was recognized. Please try again.")
        if speech_text == "thank you goodbye":
            print("Stopping program")
            break
        print(speech_text)

        intents = pred_class(speech_text, words, classes)
        result = get_response(intents, data)
        print(result)
        convert_text_to_speech(result)

    
if __name__ == '__main__':
    main()

