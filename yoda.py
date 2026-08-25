#!/usr/bin/env python3
# assistant.py
# YodaAssistant consolidado: reconhecimento de voz, TTS, detecção e cadastro/identificação de faces,
# agenda (cadastrar, mostrar, limpar), data/hora, e integração opcional Groq (IA generativa).
# Usa bibliotecas: speech_recognition, pyttsx3, gTTS, pygame, cv2

import os
import time
import json
import glob
import cv2
import numpy as np
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
from pygame import mixer
from datetime import datetime

# Groq (opcional)
try:
    from groq import Groq
    _HAS_GROQ_LIB = True
except Exception:
    Groq = None
    _HAS_GROQ_LIB = False

# Detect LBPH availability (requires opencv-contrib-python)
try:
    _HAS_LBPH = hasattr(cv2, "face") and callable(getattr(cv2.face, "LBPHFaceRecognizer_create", None))
except Exception:
    _HAS_LBPH = False

class YodaAssistant:
    def __init__(self, tts_method=None, lang="pt-BR", mic_index=None):
        # Config
        self.LANG = lang
        # Use gTTS by default; can be overridden by tts_method param or ASSISTANT_TTS env var
        self.TTS_METHOD = tts_method or os.getenv("ASSISTANT_TTS", "gtts")
        self.mic_index = mic_index

        # ASR
        self.recognizer = sr.Recognizer()
        if self.mic_index is None:
            self.microphone = sr.Microphone()
        else:
            self.microphone = sr.Microphone(device_index=self.mic_index)

        # TTS (pyttsx3 kept for offline fallback)
        self.pytt_engine = pyttsx3.init()
        self.pytt_engine.setProperty("volume", 1.0)
        self.pytt_engine.setProperty("rate", 180)
        self._mixer_inited = False

        # Wake word
        self.wake_word = "yoda"
        self.running = False

        # Faces storage
        self.faces_dir = "faces"
        os.makedirs(self.faces_dir, exist_ok=True)
        self.labels_path = os.path.join(self.faces_dir, "labels.json")
        self.model_path = os.path.join(self.faces_dir, "face_model.yml")
        self.labels = self._load_labels()

        # LBPH recognizer if available
        self.recognizer_model = None
        if _HAS_LBPH:
            try:
                self.recognizer_model = cv2.face.LBPHFaceRecognizer_create()
                if os.path.exists(self.model_path):
                    # load if exists
                    try:
                        self.recognizer_model.read(self.model_path)
                    except Exception:
                        # older OpenCV versions may use .read differently
                        pass
            except Exception as e:
                print("Erro inicializando LBPH:", e)
                self.recognizer_model = None

        # Agenda file
        self.agenda_path = "agenda.txt"
        if not os.path.exists(self.agenda_path):
            open(self.agenda_path, "a", encoding="utf-8").close()

        # Groq client (optional)
        self.groq_client = None
        groq_key = os.getenv("GROQ_API_KEY")
        if _HAS_GROQ_LIB and groq_key:
            try:
                self.groq_client = Groq(api_key=groq_key)
            except Exception as e:
                print("Erro ao inicializar Groq client:", e)
                self.groq_client = None

    # ---------------- TTS ----------------
    def tts_pyttsx3_say(self, msg: str):
        if not msg:
            return
        try:
            self.pytt_engine.say(msg)
            self.pytt_engine.runAndWait()
        except Exception as e:
            print("Erro pyttsx3:", e)

    def tts_gtts_play(self, msg: str, filename="response_br.mp3"):
        if msg is None:
            msg = ""
        # Prepara tentativas de códigos de idioma (gTTS pode aceitar 'pt' ou 'pt-br' etc.)
        lang_candidates = []
        if self.LANG:
            lang_candidates.append(self.LANG)
            # adicionar variante curta (ex.: 'pt' de 'pt-BR')
            if "-" in self.LANG:
                lang_candidates.append(self.LANG.split("-")[0])
        # garantir pt como fallback
        lang_candidates.extend(["pt", "pt-br", "pt-BR"])
        used_tts = None
        for lang_try in lang_candidates:
            try:
                tts = gTTS(msg, lang=lang_try)
                used_tts = tts
                break
            except Exception:
                used_tts = None
                continue
        if used_tts is None:
            # última tentativa sem idioma (deixa gTTS escolher) — ainda pode falhar
            try:
                used_tts = gTTS(msg)
            except Exception as e:
                print("gTTS falhou ao gerar áudio:", e)
                return
        try:
            used_tts.save(filename)
        except Exception as e:
            print("Falha ao salvar arquivo TTS:", e)
            return
        try:
            if not self._mixer_inited:
                mixer.init()
                self._mixer_inited = True
            mixer.music.load(filename)
            mixer.music.play()
            while mixer.music.get_busy():
                time.sleep(0.1)
            try:
                mixer.music.unload()
            except Exception:
                pass
        finally:
            try:
                os.remove(filename)
            except Exception:
                pass

    def falar(self, msg: str):
        if not msg:
            return
        if str(self.TTS_METHOD).lower() == "gtts":
            self.tts_gtts_play(msg)
        else:
            # mantenha pyttsx3 como fallback/offline option
            self.tts_pyttsx3_say(msg)

    # ---------------- ASR ----------------
    def ouvir_uma_vez(self, timeout=8, phrase_time_limit=6):
        with self.microphone as src:
            try:
                self.recognizer.adjust_for_ambient_noise(src, duration=0.8)
            except Exception:
                pass
            print("Escutando...")
            try:
                audio = self.recognizer.listen(src, timeout=timeout, phrase_time_limit=phrase_time_limit)
            except Exception as e:
                print("Erro captura áudio:", e)
                return None
        try:
            texto = self.recognizer.recognize_google(audio, language=self.LANG)
            print("Transcrito:", texto)
            return texto
        except sr.UnknownValueError:
            print("Não entendi o áudio.")
            return None
        except sr.RequestError as e:
            print("Erro no serviço ASR:", e)
            return None
        except Exception as e:
            print("Erro no reconhecimento:", e)
            return None

    # ---------------- Faces util ----------------
    def _load_labels(self):
        if os.path.exists(self.labels_path):
            try:
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_labels(self):
        with open(self.labels_path, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, ensure_ascii=False, indent=2)

    def _get_or_create_label_id(self, name: str):
        name = name.strip()
        if name in self.labels:
            return self.labels[name]
        used = list(self.labels.values())
        new_id = 1 if not used else max(used) + 1
        self.labels[name] = new_id
        self._save_labels()
        return new_id

    def _detect_first_face_in_frame(self, frame):
        cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None, None
        x, y, w, h = faces[0]
        face_img = gray[y:y+h, x:x+w]
        return face_img, (x, y, w, h)

    def _save_cropped_face(self, face_img, label_id):
        face_resized = cv2.resize(face_img, (200, 200))
        timestamp = int(time.time())
        filename = f"{label_id}_{timestamp}.png"
        path = os.path.join(self.faces_dir, filename)
        cv2.imwrite(path, face_resized)
        return path

    def _load_dataset_for_training(self):
        images = []
        labels = []
        for path in glob.glob(os.path.join(self.faces_dir, "*.png")):
            fname = os.path.basename(path)
            try:
                label_str = fname.split("_", 1)[0]
                label_id = int(label_str)
            except Exception:
                continue
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_resized = cv2.resize(img, (200, 200))
            images.append(img_resized)
            labels.append(label_id)
        return images, labels

    def _train_recognizer_if_possible(self):
        images, labels = self._load_dataset_for_training()
        if not images or not labels:
            return False
        if _HAS_LBPH:
            try:
                if self.recognizer_model is None:
                    self.recognizer_model = cv2.face.LBPHFaceRecognizer_create()
                import numpy as _np
                self.recognizer_model.train(images, _np.array(labels))
                self.recognizer_model.write(self.model_path)
                print("Modelo LBPH treinado e salvo.")
                return True
            except Exception as e:
                print("Falha ao treinar LBPH:", e)
                self.recognizer_model = None
                return False
        else:
            return False

    def _histogram_match_confidence(self, face_img):
        probe = cv2.resize(face_img, (200, 200))
        hist_probe = cv2.calcHist([probe], [0], None, [256], [0, 256])
        cv2.normalize(hist_probe, hist_probe)
        best_conf = -1.0
        best_name = None
        for path in glob.glob(os.path.join(self.faces_dir, "*.png")):
            fname = os.path.basename(path)
            try:
                label_id = int(fname.split("_",1)[0])
            except Exception:
                continue
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_res = cv2.resize(img, (200, 200))
            hist_db = cv2.calcHist([img_res], [0], None, [256], [0,256])
            cv2.normalize(hist_db, hist_db)
            corr = cv2.compareHist(hist_probe, hist_db, cv2.HISTCMP_CORREL)
            if corr > best_conf:
                best_conf = corr
                name = None
                for k,v in self.labels.items():
                    if v == label_id:
                        name = k
                        break
                best_name = name
        if best_conf < -0.5 or best_name is None:
            return None, 0.0
        conf_percent = float((best_conf + 1.0) / 2.0 * 100.0)
        return best_name, conf_percent

    def capture_face_and_label(self, ask_name_via_speech=True):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.falar("Não foi possível abrir a câmera.")
            return
        self.falar("Posicione a pessoa na frente da câmera. Aguarde um instante.")
        time.sleep(1.0)
        face_saved_path = None
        try:
            for _ in range(60):
                ret, frame = cap.read()
                if not ret:
                    continue
                face_img, rect = self._detect_first_face_in_frame(frame)
                if face_img is not None:
                    x, y, w, h = rect
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.imshow("Capturando - pressione qualquer tecla", frame)
                    cv2.waitKey(200)
                    if ask_name_via_speech:
                        self.falar("Quem é essa pessoa? Diga o nome agora.")
                        name = self.ouvir_uma_vez(timeout=6, phrase_time_limit=4)
                    else:
                        name = None
                    if not name or not name.strip():
                        try:
                            name = input("Informe o nome da pessoa (fallback): ").strip()
                        except Exception:
                            name = None
                    if not name:
                        self.falar("Nome inválido. Cancelando captura.")
                        break
                    label_id = self._get_or_create_label_id(name)
                    saved = self._save_cropped_face(face_img, label_id)
                    face_saved_path = saved
                    self.falar(f"Rótulo salvo para {name}.")
                    self._train_recognizer_if_possible()
                    break
            if face_saved_path is None:
                self.falar("Não detectei nenhuma face. Tente novamente.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return face_saved_path

    def identify_face_from_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.falar("Não foi possível abrir a câmera.")
            return None, 0.0
        self.falar("Aguardando uma face para identificação. Posicione a pessoa.")
        name = None
        confidence = 0.0
        try:
            for _ in range(80):
                ret, frame = cap.read()
                if not ret:
                    continue
                face_img, rect = self._detect_first_face_in_frame(frame)
                if face_img is not None:
                    face_resized = cv2.resize(face_img, (200,200))
                    if self.recognizer_model is not None:
                        try:
                            label_id, conf = self.recognizer_model.predict(face_resized)
                            conf_percent = max(0.0, min(100.0, 100.0 - (conf / 10.0)))
                            name = None
                            for k,v in self.labels.items():
                                if v == label_id:
                                    name = k
                                    break
                            confidence = conf_percent
                        except Exception as e:
                            print("Erro predizendo LBPH:", e)
                            name, confidence = self._histogram_match_confidence(face_resized)
                    else:
                        name, confidence = self._histogram_match_confidence(face_resized)
                    if rect:
                        x,y,w,h = rect
                        cv2.rectangle(frame, (x,y),(x+w,y+h), (0,255,0),2)
                        cv2.putText(frame, f"{name or 'Desconhecido'} {confidence:.0f}%", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.imshow("Identificação - pressione qualquer tecla", frame)
                    cv2.waitKey(800)
                    break
            if name is None:
                self.falar("Não reconheci essa pessoa.")
            else:
                self.falar(f"Eu acho que é {name} com {int(confidence)} por cento de confiança.")
            return name, confidence
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def camera_detect_stream(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector_facial = cv2.CascadeClassifier(cascade_path)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.falar("Não foi possível abrir a câmera.")
            return
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector_facial.detectMultiScale(gray, 1.1, 5)
                for (x, y, w, h) in faces:
                    # crop e preparar face para reconhecimento
                    face_gray = gray[y:y + h, x:x + w]
                    try:
                        face_resized = cv2.resize(face_gray, (200, 200))
                    except Exception:
                        face_resized = cv2.resize(face_gray, (100, 100))
                    name = "Desconhecido"
                    confidence = 0.0
                    # tenta LBPH se disponível e treinado
                    if self.recognizer_model is not None:
                        try:
                            label_id, conf = self.recognizer_model.predict(face_resized)
                            # converte score do recognizer para percentagem (heurística usada no resto do código)
                            conf_percent = max(0.0, min(100.0, 100.0 - (conf / 10.0)))
                            # encontra nome pelo label
                            for k, v in self.labels.items():
                                if v == label_id:
                                    name = k
                                    break
                            confidence = conf_percent
                        except Exception as e:
                            # fallback por histograma caso LBPH falhe
                            print("Erro predizendo LBPH no stream:", e)
                            name, confidence = self._histogram_match_confidence(face_resized)
                    else:
                        # sem LBPH: usar comparação por histograma (pode ser mais lento)
                        name, confidence = self._histogram_match_confidence(face_resized)

                    # desenha retângulo e texto com nome+confiança
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label_text = f"{name or 'Desconhecido'} {int(confidence)}%"
                    # posiciona texto acima do retângulo (cuida de caso y<20)
                    text_y = y - 10 if y - 10 > 10 else y + 20
                    cv2.putText(frame, label_text, (x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Detecção de Faces - Yoda", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    # ---------------- Agenda ----------------
    def cadastrar_evento_via_voz(self):
        self.falar("Claro. Diga o título ou descrição do evento.")
        descricao = self.ouvir_uma_vez(timeout=8, phrase_time_limit=8)
        if not descricao:
            self.falar("Não recebi a descrição. Cancelando cadastro.")
            return False
        self.falar("Agora, diga a data e hora do evento, por exemplo: dia 25 de dezembro às 15 horas, ou diga apenas 'sem data' para manter sem data.")
        data_hora = self.ouvir_uma_vez(timeout=8, phrase_time_limit=8)
        if not data_hora:
            data_hora = "sem data"
        timestamp_salvo = datetime.utcnow().isoformat() + "Z"
        linha = f"{timestamp_salvo} | {data_hora.strip()} | {descricao.strip()}\n"
        try:
            with open(self.agenda_path, "a", encoding="utf-8") as f:
                f.write(linha)
            self.falar("Evento cadastrado na agenda.")
            return True
        except Exception as e:
            print("Erro ao salvar evento:", e)
            self.falar("Falha ao salvar o evento.")
            return False

    def mostrar_agenda(self, max_items=10):
        try:
            with open(self.agenda_path, "r", encoding="utf-8") as f:
                linhas = [l.strip() for l in f.readlines() if l.strip()]
        except Exception as e:
            print("Erro lendo agenda:", e)
            self.falar("Não consegui acessar a agenda.")
            return
        if not linhas:
            self.falar("Sua agenda está vazia.")
            return
        for linha in linhas[-max_items:]:
            partes = [p.strip() for p in linha.split("|")]
            if len(partes) >= 3:
                ts_salvo, data_hora, descricao = partes[0], partes[1], "|".join(partes[2:])
            elif len(partes) == 2:
                ts_salvo, data_hora = partes
                descricao = ""
            else:
                descricao = partes[0]
                data_hora = "sem data"
                ts_salvo = ""
            leg_ts = ts_salvo
            try:
                leg = datetime.fromisoformat(ts_salvo.replace("Z", "+00:00"))
                leg_ts = leg.strftime("%d/%m/%Y %H:%M")
            except Exception:
                pass
            frase = f"{descricao}. Data informada: {data_hora}. Registrado em: {leg_ts}."
            print("Agenda item:", frase)
            self.falar(frase)
            time.sleep(0.4)

    def clear_agenda(self, require_confirmation=True):
        if require_confirmation:
            self.falar("Tem certeza que deseja apagar todos os eventos da agenda? Diga 'sim' para confirmar ou 'não' para cancelar.")
            resp = self.ouvir_uma_vez(timeout=6, phrase_time_limit=3)
            if not resp:
                self.falar("Não recebi confirmação. Operação cancelada.")
                return False
            if resp.strip().lower() not in ["sim", "s", "confirmar", "yes", "claro"]:
                self.falar("Operação cancelada.")
                return False
        try:
            with open(self.agenda_path, "w", encoding="utf-8") as f:
                f.truncate(0)
            self.falar("Agenda apagada.")
            return True
        except Exception as e:
            print("Erro ao limpar agenda:", e)
            self.falar("Falha ao apagar a agenda.")
            return False

    # ---------------- Date/time ----------------
    def obter_data_e_hora_atual(self):
        meses = [
            "janeiro", "fevereiro", "março", "abril", "maio", "junho",
            "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"
        ]
        agora = datetime.now()
        dia = agora.day
        mes = meses[agora.month - 1]
        hora = agora.hour
        minuto = agora.minute
        hora_str = f"{hora} hora" + ("s" if hora != 1 else "")
        minuto_str = f"{minuto} minuto" + ("s" if minuto != 1 else "")
        if minuto == 0:
            tempo = f"{hora_str}"
        else:
            tempo = f"{hora_str} e {minuto_str}"
        return f"Hoje é {dia} de {mes} e {tempo}."

    # ---------------- Groq integration ----------------
    def gerar_resposta_groq(self, prompt: str, model: str = "openai/gpt-oss-120b"):
        if not _HAS_GROQ_LIB:
            return "A biblioteca 'groq' não está instalada no ambiente."
        if not self.groq_client:
            return "Chave da API Groq não configurada (GROQ_API_KEY)."
        try:
            t0 = time.time()
            response = self.groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Você é um assistente útil e objetivo."},
                    {"role": "user", "content": prompt}
                ]
            )
            t1 = time.time()
            content = None
            try:
                content = response.choices[0].message.content
            except Exception:
                content = str(response)
            print(f"Tempo da requisição Groq: {t1 - t0:.2f}s")
            return content
        except Exception as e:
            print("Erro ao chamar Groq:", e)
            return f"Erro ao chamar a API de IA: {e}"

    # ---------------- Command processing ----------------
    def processar_comando(self, texto: str):
        if not texto or not texto.strip():
            self.falar("Sim?")
            cmd = self.ouvir_uma_vez(timeout=6, phrase_time_limit=6)
            if cmd:
                self.processar_comando(cmd)
            return

        texto_lower = texto.lower()

        # encerrar
        if any(p in texto_lower for p in ["tchau", "adeus", "sair", "encerrar", "pare"]):
            self.falar("Encerrando. Até logo.")
            self.running = False
            return

        # data/hora
        if any(p in texto_lower for p in ["que dia é hoje", "qual a data", "dia e mês", "que dia", "data de hoje", "qual é a data"]):
            resposta = self.obter_data_e_hora_atual()
            print("Yoda (data/hora):", resposta)
            self.falar(resposta)
            return

        if any(p in texto_lower for p in ["hora", "que horas", "horas", "horário"]):
            resp = f"Agora são {datetime.now().hour} horas e {datetime.now().minute} minutos."
            print("Yoda (hora):", resp)
            self.falar(resp)
            return

        # Agenda commands
        if any(p in texto_lower for p in ["cadastrar evento", "adicionar evento", "salvar evento", "criar evento", "adicionar na agenda", "cadastrar na agenda"]):
            success = self.cadastrar_evento_via_voz()
            if success:
                self.falar("Evento salvo com sucesso.")
            return

        if any(p in texto_lower for p in ["mostrar agenda", "exibir agenda", "ver agenda", "mostrar minha agenda"]):
            self.falar("Mostrando sua agenda agora.")
            self.mostrar_agenda()
            return

        if any(p in texto_lower for p in ["apagar agenda", "limpar agenda", "resetar agenda", "apagar os eventos", "limpar os eventos"]):
            ok = self.clear_agenda(require_confirmation=True)
            return

        # Face commands
        if any(p in texto_lower for p in ["salvar face", "gravar face", "ensinar pessoa", "salvar pessoa", "gravar pessoa"]):
            self.falar("Capturando face para salvar.")
            self.capture_face_and_label(ask_name_via_speech=True)
            return

        if "treinar" in texto_lower and "face" in texto_lower:
            ok = self._train_recognizer_if_possible()
            if ok:
                self.falar("Treinei o modelo de reconhecimento de faces.")
            else:
                self.falar("Não foi possível treinar (talvez não haja dados ou LBPH não esteja disponível).")
            return

        if any(p in texto_lower for p in ["quem é", "quem é essa pessoa", "quem é essa", "quem é ele", "quem é ela"]):
            self.falar("Tentarei identificar agora.")
            self.identify_face_from_camera()
            return

        # Camera stream
        if any(p in texto_lower for p in ["camera", "câmera", "mostrar câmera", "mostrar camera", "detectar face"]):
            self.falar("Abrindo câmera para detecção. Pressione 'q' para sair.")
            self.camera_detect_stream()
            return

        # cálculo
        if "calcular" in texto_lower:
            tokens = texto.split()
            try:
                idx = [i for i,w in enumerate(tokens) if w.lower()=="calcular"][0]
                if len(tokens) > idx + 3:
                    a = int(tokens[idx+1]); op = tokens[idx+2]; b = int(tokens[idx+3])
                    if op == "+":
                        self.falar(f"O resultado é {a+b}")
                    elif op in ("x","*"):
                        self.falar(f"O resultado é {a*b}")
                    elif op == "-":
                        self.falar(f"O resultado é {a-b}")
                    elif op == "/":
                        self.falar(f"O resultado é {a/b}")
                    else:
                        self.falar("Operador não reconhecido.")
                else:
                    self.falar("Formato de cálculo inválido. Diga: calcular 3 + 4")
            except Exception as e:
                self.falar(f"Erro ao calcular: {e}")
            return

        # fallback -> generative IA se disponível
        if self.groq_client:
            prompt = texto.strip()
            self.falar("Consultando a IA, aguarde...")
            resposta_ia = self.gerar_resposta_groq(prompt)
            print("Yoda (IA):", resposta_ia)
            self.falar(resposta_ia)
            return

        # default fallback
        self.falar("Comando não reconhecido.")

    # ---------------- handler/run ----------------
    def handle_transcription(self, texto: str):
        if not texto:
            return
        texto_lower = texto.lower()
        if self.wake_word in texto_lower:
            parts = texto_lower.split(self.wake_word, 1)
            after = parts[1].strip(" ,:.-")
            if after:
                self.processar_comando(after)
            else:
                self.falar("Sim?")
                cmd = self.ouvir_uma_vez(timeout=6, phrase_time_limit=6)
                if cmd:
                    self.processar_comando(cmd)
        else:
            print("Wake word não detectada — aguardando 'Yoda'...")

    def run(self):
        self.running = True
        self.falar("Yoda iniciado. Diga 'Yoda' seguido do comando.")
        try:
            while self.running:
                texto = self.ouvir_uma_vez()
                if texto:
                    self.handle_transcription(texto)
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("Encerrado pelo usuário.")
        finally:
            print("Yoda encerrado.")


if __name__ == "__main__":
    assistant = YodaAssistant()
    assistant.run()
