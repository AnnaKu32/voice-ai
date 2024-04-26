from gtts import gTTS
import os

def convert_text_to_speech(text):
    myobj = gTTS(text=text, lang='en', slow=False)
    myobj.save("response.mp3")
    os.system("start response.mp3")