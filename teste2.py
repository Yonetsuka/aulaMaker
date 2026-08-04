from gtts import gTTS
from pygame import mixer
import datetime

audio = gTTS(f"Agora são {datetime.datetime.now().hour} horas e {datetime.datetime.now().minute}minutos", lang="pt")

audio.save("mensagem.mp3")

mixer.init()
mixer.music.load("mensagem.mp3")
mixer.music.play()

while mixer.music.get_busy():
    continue