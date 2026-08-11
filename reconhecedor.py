import speech_recognition as sr

reconhecedor = sr.Recognizer()

with sr.Microphone() as mic:
    reconhecedor.adjust_for_ambient_noise(mic, duration=1)
    print("Fale algo...")
    audio = reconhecedor.listen(mic)
    print("Processando áudio")
    texto = reconhecedor.recognize_google(audio, language='pt-BR')
    print(f"você falou {texto}")

    try:
        if "calcular" in texto.lower():
            lista = texto.split()
            print(lista)
            if lista[2] == "+":
                print(f"O resultado é {int(lista[1]) + int(lista[3])}")
            elif lista[2] == "x":
                print(f"O resultado é {int(lista[1]) * int(lista[3])}")
            elif lista[2] == "-":
                print(f"O resultado é {int(lista[1]) - int(lista[3])}")
            elif lista[2] == "/":
                print(f"O resultado é {int(lista[1]) / int(lista[-1])}")
    except Exception as e:
        print(e)

