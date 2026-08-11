from gtts import gTTS
from pygame import mixer

audio = gTTS("", lang='pt-BR')
audio.save("brasil.mp3")

mixer.init()
mixer.music.load("brasil.mp3")
mixer.music.play()

while mixer.music.get_busy():
    continue

mixer.music.unload()