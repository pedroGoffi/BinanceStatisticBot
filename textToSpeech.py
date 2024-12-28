import pyttsx3


class TTS:
    engine: pyttsx3.Engine
    def __init__(self):
        print("... Iniciando engine de voz ...")
        self.engine = pyttsx3.init()
        self.engine.voice = self.engine.getProperty('voices')[0]
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1)


    def log(self, text: str):        
        self.engine.say(text)
        self.engine.runAndWait()
        
        