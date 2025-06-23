import json
from vosk import Model, KaldiRecognizer
import spacy
import pyaudio
import noisereduce as nr
import numpy as np
import os
import requests
import zipfile
from googletrans import Translator
import threading
import subprocess

LANG_MODELS = {
    'en': 'models/vosk-model-en-us-0.42-gigaspeech',
    'hi': 'models/vosk-model-hi-0.22',
    'zh': 'models/vosk-model-cn-0.22',
    'ru': 'models/vosk-model-ru-0.42',
    'fr': 'models/vosk-model-fr-0.22',
    'de': 'models/vosk-model-de-0.21',
    'es': 'models/vosk-model-es-0.42',
    'pt': 'models/vosk-model-pt-fb-v0.1.1-20220516_2113',
    'ja': 'models/vosk-model-ja-0.22',
    'it': 'models/vosk-model-it-0.22',
}

NLP_MODELS = {
    'en': 'en_core_web_sm',
    'hi': 'xx_ent_wiki_sm',  
    'zh': 'zh_core_web_sm',
    'ru': 'ru_core_news_sm',
    'fr': 'fr_core_news_sm',
    'de': 'de_core_news_sm',
    'es': 'es_core_news_sm',
    'pt': 'pt_core_news_sm',
    'ja': 'ja_core_news_sm',
    'it': 'it_core_news_sm',
}

MODEL_URLS = {
    'hi': 'https://alphacephei.com/vosk/models/vosk-model-hi-0.22.zip',
    'zh': 'https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip',
    'ru': 'https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip',
    'fr': 'https://alphacephei.com/vosk/models/vosk-model-fr-0.22.zip',
    'de': 'https://alphacephei.com/vosk/models/vosk-model-de-0.21.zip',
    'es': 'https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip',
    'pt': 'https://alphacephei.com/vosk/models/vosk-model-pt-fb-v0.1.1-20220516_2113.zip',
    'ja': 'https://alphacephei.com/vosk/models/vosk-model-ja-0.22.zip',
    'it': 'https://alphacephei.com/vosk/models/vosk-model-it-0.22.zip',
    # English is always present locally
}

def ensure_model_downloaded(lang):
    if lang == 'en':
        return  # English model is always present
    model_path = LANG_MODELS[lang]
    if not os.path.exists(model_path):
        print(f"Model for '{lang}' not found. Downloading...")
        url = MODEL_URLS[lang]
        zip_path = f"{model_path}.zip"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(model_path))
        os.remove(zip_path)
        print(f"Model for '{lang}' downloaded and extracted.")
    else:
        print(f"Model for '{lang}' already exists.")

class ASR:
    def __init__(self, lang='en'):
        ensure_model_downloaded(lang)
        # Ensure spaCy model is installed
        nlp_model = NLP_MODELS[lang]
        try:
            self.nlp = spacy.load(nlp_model)
        except OSError:
            print(f"spaCy model '{nlp_model}' not found. Downloading...")
            subprocess.run(["python", "-m", "spacy", "download", nlp_model], check=True)
            self.nlp = spacy.load(nlp_model)
        self.model = Model(LANG_MODELS[lang])
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.lang = lang
        self.translator = Translator()

    def listen(self):
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            print(p.get_device_info_by_index(i))

        stream = p.open(rate=16000, channels=1, format=pyaudio.paInt16, 
                        input=True, output=False, frames_per_buffer=8192)
        stream.start_stream()
        final_text = ""
        try:
            while True:
                try:
                    data = stream.read(8192, exception_on_overflow=False)
                    if len(data) == 0:
                        break
                    # Real-time noise reduction
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    reduced_chunk = nr.reduce_noise(y=audio_chunk, sr=16000, stationary=True, prop_decrease=1.0)
                    data = reduced_chunk.tobytes()
                    if self.recognizer.AcceptWaveform(data):
                        res = json.loads(self.recognizer.Result())
                        print(res["text"])
                        final_text = res["text"]
                        if res.get("text", "").lower() == "stop listening":
                            break
                    else:
                        partial_res = json.loads(self.recognizer.PartialResult())
                        if partial_res.get("partial"):
                            print("Partial:", partial_res["partial"])
                except OSError as e:
                    print(f"PyAudio error: {e}")
                except Exception as e:
                    print(f"ASR error: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
        # Translate if not English
        translated = final_text
        if self.lang != 'en' and final_text:
            try:
                translated = self.translator.translate(final_text, src=self.lang, dest='en').text
                print(f"Translated to English: {translated}")
            except Exception as e:
                print(f"Translation error: {e}")
        return final_text, translated

def multi_asr_listen(langs):
    results = {}
    threads = []
    def run_asr(lang):
        asr = ASR(lang)
        original, translated = asr.listen()
        results[lang] = (original, translated)
    for lang in langs:
        t = threading.Thread(target=run_asr, args=(lang,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    # Choose the most relevant (longest non-empty English translation)
    best_english = max((v[1] for v in results.values()), key=lambda s: len(s.strip()) if s else 0, default="")
    return {"all_results": results, "best_english": best_english}
