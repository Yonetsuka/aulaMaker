import pyttsx3

engine = pyttsx3.init()
engine.setProperty('volume', 1.0)
engine.setProperty('rate', 200)

msg = "six seven"
engine.say(msg)
engine.runAndWait()