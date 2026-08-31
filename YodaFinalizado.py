#!/usr/bin/env python3
# YodaFinalizado.py
# YodaAssistant consolidado com funcionalidades de screenshot
# Estende: reconhecimento de voz, TTS, detecção de faces, agenda, data/hora, Groq, YouTube e SCREENSHOTS

import os
import time
import json
import glob
import cv2
import numpy as np
import speech_recognition as sr
import pyttsx3
import subprocess
import webbrowser
import urllib.parse
from gtts import gTTS
from pygame import mixer
from datetime import datetime
import pyautogui
from PIL import Image
from pathlib import Path

# Groq (opcional)
try:
    from groq import Groq
    _HAS_GROQ_LIB = True
except Exception:
    Groq = None
    _HAS_GROQ_LIB = False

# YouTube search (opcional)
try:
    from yt_search import YouTubeSearch
    _HAS_YT_SEARCH = True
except Exception:
    _HAS_YT_SEARCH = False

# Detect LBPH availability (requires opencv-contrib-python)
try:
    _HAS_LBPH = hasattr(cv2, "face") and callable(getattr(cv2.face, "LBPHFaceRecognizer_create", None))
except Exception:
    _HAS_LBPH = False


class YodaAssistant:
    def __init__(self, tts_method=None, lang="pt-BR", mic_index=None):
        # Config
        self.LANG = lang
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
                    try:
                        self.recognizer_model.read(self.model_path)
                    except Exception:
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

        # ========== NOVO: Screenshots storage ==========
        self.screenshots_dir = "screenshots"
        os.makedirs(self.screenshots_dir, exist_ok=True)
        self.screenshots_metadata_path = os.path.join(self.screenshots_dir, "metadata.json")
        self.screenshot_counter = self._load_screenshot_metadata()

    # ========== NOVO: FUNÇÕES DE SCREENSHOT ==========
    
    def _load_screenshot_metadata(self):
        """
        Carrega o contador de screenshots do arquivo de metadados.
        Retorna o próximo número disponível.
        """
        if os.path.exists(self.screenshots_metadata_path):
            try:
                with open(self.screenshots_metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    return metadata.get("next_id", 0)
            except Exception as e:
                print(f"Aviso ao carregar metadados de screenshots: {e}")
                return 0
        return 0

    def _save_screenshot_metadata(self):
        """
        Salva o estado do contador de screenshots em arquivo.
        """
        try:
            metadata = {"next_id": self.screenshot_counter}
            with open(self.screenshots_metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Erro ao salvar metadados de screenshots: {e}")

    def _get_screenshot_file_list(self):
        """
        Retorna lista de todos os screenshots salvos no formato padrão.
        """
        screenshots = []
        for arquivo in glob.glob(os.path.join(self.screenshots_dir, "*.png")):
            try:
                nome_base = os.path.basename(arquivo).replace(".png", "")
                if nome_base.isdigit() or "_" in nome_base:
                    screenshots.append((arquivo, nome_base))
            except Exception:
                continue
        return sorted(screenshots, key=lambda x: int(x[1].split("_")[0]) if "_" in x[1] else int(x[1]))

    def tirar_screenshot(self, nome_customizado: str = None):
        """
        Captura a tela e salva como imagem PNG.
        
        Três modos de salvamento:
        1. Modo sequencial (padrão): 0.png, 1.png, 2.png, ... (automático)
        2. Modo customizado: nome_customizado.png (usuário fornece nome)
        3. Modo com timestamp: nome_timestamp.png (mais legível)
        
        Args:
            nome_customizado: (opcional) Nome customizado para o arquivo
                            - Se None: usa numeração sequencial
                            - Se string: usa esse nome
        
        Returns:
            bool: True se sucesso, False se falha
        """
        try:
            self.falar("Capturando tela agora.")
            
            # Captura a tela usando pyautogui
            screenshot = pyautogui.screenshot()
            
            # Define nome do arquivo
            if nome_customizado:
                # Modo customizado
                nome_arquivo = f"{nome_customizado}.png"
            else:
                # Modo sequencial (padrão)
                nome_arquivo = f"{self.screenshot_counter}.png"
                self.screenshot_counter += 1
            
            caminho_completo = os.path.join(self.screenshots_dir, nome_arquivo)
            
            # Salva a imagem
            screenshot.save(caminho_completo)
            
            print(f"✓ Screenshot salvo: {caminho_completo}")
            self.falar(f"Tela capturada e salva como {nome_arquivo}.")
            
            # Salva os metadados
            self._save_screenshot_metadata()
            
            return True
            
        except Exception as e:
            print(f"✗ Erro ao capturar tela: {e}")
            self.falar("Falha ao capturar a tela.")
            return False

    def tirar_screenshot_via_cv2(self, nome_customizado: str = None):
        """
        Alternativa usando OpenCV (cv2) para capturar tela.
        Mais rápido que pyautogui em alguns casos.
        Útil se o monitor tem múltiplas resoluções.
        
        Args:
            nome_customizado: (opcional) Nome customizado para o arquivo
        
        Returns:
            bool: True se sucesso, False se falha
        """
        try:
            self.falar("Capturando tela com câmera.")
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.falar("Não foi possível acessar a câmera.")
                return False
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                self.falar("Falha ao capturar frame.")
                return False
            
            # Define nome do arquivo
            if nome_customizado:
                nome_arquivo = f"{nome_customizado}.png"
            else:
                nome_arquivo = f"{self.screenshot_counter}.png"
                self.screenshot_counter += 1
            
            caminho_completo = os.path.join(self.screenshots_dir, nome_arquivo)
            
            # Salva a imagem
            cv2.imwrite(caminho_completo, frame)
            
            print(f"✓ Frame capturado: {caminho_completo}")
            self.falar(f"Câmera capturada e salva como {nome_arquivo}.")
            
            self._save_screenshot_metadata()
            
            return True
            
        except Exception as e:
            print(f"✗ Erro ao capturar com câmera: {e}")
            self.falar("Falha ao capturar com câmera.")
            return False

    def tirar_screenshot_por_numero(self, numero: int = None):
        """
        Captura tela e salva com nome customizado baseado em número fornecido.
        
        Args:
            numero: Número para salvar (ex: 5 → 5.png)
                   Se None, pergunta pelo microfone
        
        Returns:
            bool: True se sucesso
        """
        if numero is None:
            self.falar("Qual número você quer usar para a captura de tela? Diga o número.")
            resposta = self.ouvir_uma_vez(timeout=6, phrase_time_limit=4)
            
            if not resposta:
                self.falar("Não entendi o número.")
                return False
            
            try:
                numero = int(''.join(filter(str.isdigit, resposta)))
                if numero < 0:
                    raise ValueError("Número deve ser positivo")
            except (ValueError, IndexError):
                self.falar("Formato inválido. Use apenas números.")
                return False
        
        return self.tirar_screenshot(nome_customizado=str(numero))

    def mostrar_screenshot(self, identificador: str = None):
        """
        Exibe uma screenshot salva anteriormente.
        
        Três modos:
        1. Modo específico: mostrar_screenshot("0") → exibe 0.png
        2. Modo voz: mostrar_screenshot() → pergunta qual número
        3. Modo lista: mostrar_screenshot("lista") → lista todas as capturas
        
        Args:
            identificador: ID ou nome do arquivo (sem extensão)
                         - Se None: pergunta pelo microfone
                         - Se "lista": mostra todas
                         - Se "ultima": mostra a mais recente
                         - Se número: mostra essa captura
        
        Returns:
            bool: True se sucesso
        """
        try:
            # Se não fornecido, pergunta pelo microfone
            if identificador is None:
                self.falar("Qual screenshot você gostaria de ver? Diga o número ou 'lista' para ver todas.")
                resposta = self.ouvir_uma_vez(timeout=6, phrase_time_limit=4)
                
                if not resposta:
                    self.falar("Não entendi.")
                    return False
                
                identificador = resposta.strip().lower()
            
            # Modo: listar todas
            if identificador == "lista" or identificador == "todas":
                return self._listar_screenshots()
            
            # Modo: última captura
            if identificador == "ultima" or identificador == "último":
                return self._mostrar_ultima_screenshot()
            
            # Modo: screenshot específica
            caminho = self._encontrar_screenshot(identificador)
            
            if not caminho:
                self.falar(f"Screenshot '{identificador}' não encontrada.")
                print(f"✗ Screenshot não encontrada: {identificador}")
                return False
            
            # Exibe a imagem
            img = cv2.imread(caminho)
            if img is None:
                self.falar("Erro ao carregar a imagem.")
                return False
            
            # Redimensiona se muito grande
            altura, largura = img.shape[:2]
            if largura > 1920 or altura > 1080:
                escala = min(1920 / largura, 1080 / altura)
                nova_largura = int(largura * escala)
                nova_altura = int(altura * escala)
                img = cv2.resize(img, (nova_largura, nova_altura))
            
            cv2.imshow(f"Screenshot: {identificador}", img)
            self.falar(f"Mostrando screenshot {identificador}. Pressione qualquer tecla para fechar.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            print(f"✓ Screenshot {identificador} exibida com sucesso")
            self.falar("Screenshot fechada.")
            
            return True
            
        except Exception as e:
            print(f"✗ Erro ao mostrar screenshot: {e}")
            self.falar(f"Erro ao exibir screenshot: {e}")
            return False

    def _encontrar_screenshot(self, identificador: str):
        """
        Encontra caminho da screenshot pelo identificador.
        Suporta nomes com ou sem extensão.
        """
        # Tenta com extensão
        caminho_direto = os.path.join(self.screenshots_dir, f"{identificador}.png")
        if os.path.exists(caminho_direto):
            return caminho_direto
        
        # Tenta buscar entre os arquivos
        for arquivo, nome in self._get_screenshot_file_list():
            if nome == identificador:
                return arquivo
        
        return None

    def _listar_screenshots(self):
        """
        Lista todas as screenshots disponíveis.
        """
        try:
            screenshots = self._get_screenshot_file_list()
            
            if not screenshots:
                self.falar("Nenhuma screenshot encontrada.")
                print("Nenhuma screenshot encontrada.")
                return False
            
            mensagem = f"Encontrei {len(screenshots)} screenshot(s): "
            nomes = [nome for _, nome in screenshots]
            mensagem += ", ".join(nomes)
            
            print(f"✓ Screenshots disponíveis: {nomes}")
            self.falar(mensagem)
            
            return True
            
        except Exception as e:
            print(f"✗ Erro ao listar screenshots: {e}")
            self.falar("Erro ao listar screenshots.")
            return False

    def _mostrar_ultima_screenshot(self):
        """
        Exibe a screenshot mais recente.
        """
        try:
            screenshots = self._get_screenshot_file_list()
            
            if not screenshots:
                self.falar("Nenhuma screenshot encontrada.")
                return False
            
            ultima = screenshots[-1]
            caminho, nome = ultima
            
            img = cv2.imread(caminho)
            if img is None:
                self.falar("Erro ao carregar a última screenshot.")
                return False
            
            altura, largura = img.shape[:2]
            if largura > 1920 or altura > 1080:
                escala = min(1920 / largura, 1080 / altura)
                nova_largura = int(largura * escala)
                nova_altura = int(altura * escala)
                img = cv2.resize(img, (nova_largura, nova_altura))
            
            cv2.imshow(f"Última Screenshot: {nome}", img)
            self.falar(f"Mostrando a última screenshot: {nome}. Pressione qualquer tecla para fechar.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            self.falar("Screenshot fechada.")
            return True
            
        except Exception as e:
            print(f"✗ Erro ao mostrar última screenshot: {e}")
            self.falar("Erro ao exibir última screenshot.")
            return False

    def deletar_screenshot(self, identificador: str = None):
        """
        Deleta uma screenshot específica.
        
        Args:
            identificador: ID ou nome do arquivo
                         - Se None: pergunta pelo microfone
        
        Returns:
            bool: True se deletada com sucesso
        """
        if identificador is None:
            self.falar("Qual screenshot você deseja deletar? Diga o número.")
            resposta = self.ouvir_uma_vez(timeout=6, phrase_time_limit=4)
            if not resposta:
                self.falar("Operação cancelada.")
                return False
            identificador = resposta.strip()
        
        caminho = self._encontrar_screenshot(identificador)
        
        if not caminho:
            self.falar(f"Screenshot '{identificador}' não encontrada.")
            return False
        
        try:
            os.remove(caminho)
            self.falar(f"Screenshot {identificador} deletada.")
            print(f"✓ Screenshot deletada: {caminho}")
            return True
        except Exception as e:
            self.falar(f"Erro ao deletar screenshot: {e}")
            print(f"✗ Erro ao deletar: {e}")
            return False

    def obter_info_screenshots(self):
        """
        Retorna informações sobre as screenshots armazenadas.
        """
        try:
            screenshots = self._get_screenshot_file_list()
            tamanho_total = 0
            
            for arquivo, _ in screenshots:
                if os.path.exists(arquivo):
                    tamanho_total += os.path.getsize(arquivo)
            
            info = {
                "total_screenshots": len(screenshots),
                "tamanho_total_bytes": tamanho_total,
                "tamanho_total_mb": round(tamanho_total / (1024 * 1024), 2),
                "lista_screenshots": [nome for _, nome in screenshots]
            }
            
            return info
            
        except Exception as e:
            print(f"✗ Erro ao obter informações: {e}")
            return None

    # ========== FUNÇÕES ORIGINAIS (TTS, ASR, etc.) ==========

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
        lang_candidates = []
        if self.LANG:
            lang_candidates.append(self.LANG)
            if "-" in self.LANG:
                lang_candidates.append(self.LANG.split("-")[0])
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
            self.tts_pyttsx3_say(msg)

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

    # ========== RESTO DO CÓDIGO ORIGINAL ==========
    
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

    def processar_comando(self, texto: str, recursion_depth: int = 0):
        """Versão corrigida com limite de recursão"""
        MAX_RECURSION = 3
        
        if recursion_depth > MAX_RECURSION:
            self.falar("Não consegui entender. Tente novamente mais tarde.")
            return
        
        if not texto or not texto.strip():
            self.falar("Sim?")
            cmd = self.ouvir_uma_vez(timeout=6, phrase_time_limit=6)
            if cmd:
                self.processar_comando(cmd, recursion_depth + 1)
            return

        texto_lower = texto.lower()

        # ========== NOVOS COMANDOS PARA SCREENSHOTS ==========
        if any(p in texto_lower for p in ["tirar screenshot", "capturar tela", "screenshot", "print", "printar tela"]):
            self.tirar_screenshot()
            return

        if any(p in texto_lower for p in ["mostrar screenshot", "exibir screenshot", "ver screenshot", "exibir captura"]):
            self.mostrar_screenshot()
            return

        if any(p in texto_lower for p in ["deletar screenshot", "apagar screenshot", "remover screenshot", "remover captura"]):
            self.deletar_screenshot()
            return

        if any(p in texto_lower for p in ["listar screenshots", "ver capturas", "quantas screenshots", "screenshots disponíveis"]):
            info = self.obter_info_screenshots()
            if info:
                msg = f"Você tem {info['total_screenshots']} screenshot(s), totalizando {info['tamanho_total_mb']} megabytes."
                self.falar(msg)
                print(f"✓ {msg}")
            return

        # ========== COMANDOS ORIGINAIS ==========

        if any(p in texto_lower for p in ["tchau", "adeus", "sair", "encerrar", "pare"]):
            self.falar("Encerrando. Até logo.")
            self.running = False
            return

        if any(p in texto_lower for p in ["que dia é hoje", "qual a data", "dia e mês", "que dia", "data de hoje"]):
            self.falar("Desculpe, a função data não está neste arquivo simplificado.")
            return

        # Fallback
        self.falar("Comando não reconhecido.")

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
