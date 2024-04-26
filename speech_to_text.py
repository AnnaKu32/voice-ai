import speech_recognition as sr
import pyaudio

def initialize_recognizer():
    return sr.Recognizer()

def list_audio_devices():
    p = pyaudio.PyAudio()
    # for i in range(p.get_device_count()):
    #     print(p.get_device_info_by_index(i))

def capture_audio(recognizer, device_index=1, timeout=None, phrase_time_limit=10):
    with sr.Microphone(device_index) as source:
        recognizer.adjust_for_ambient_noise(source, 0.5)
        print("Say something")
        audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    return audio

def recognize_speech(recognizer, audio, language='english'):
    try:
        return recognizer.recognize_google(audio, language=language)
    except Exception:
        print("Something went wrong")
        return None