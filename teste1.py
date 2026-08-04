import pyttsx3

sintetizador = pyttsx3.init()

sintetizador.setProperty("rate",500)
sintetizador.setProperty("volume",1)
sintetizador.setProperty("voice",r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0")

msg = ""

sintetizador.say(msg)
sintetizador.runAndWait()

for voz in sintetizador.getProperty('voices'):
    print(voz)