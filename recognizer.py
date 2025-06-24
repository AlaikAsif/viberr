import json
import pyaudio
import numpy as np
import noisereduce as nr
import threading
from vosk import Model, KaldiRecognizer
from translate_util import translate_to_english

# Global registry for running ASR instances
_running_asr_instances = []
_running_asr_lock = threading.Lock()

def register_asr_instance(instance):
    with _running_asr_lock:
        _running_asr_instances.append(instance)

def unregister_asr_instance(instance):
    with _running_asr_lock:
        if instance in _running_asr_instances:
            _running_asr_instances.remove(instance)

def terminate_all_asr_instances():
    with _running_asr_lock:
        for instance in list(_running_asr_instances):
            try:
                instance.terminate()
            except Exception as e:
                print(f"Error terminating ASR instance: {e}")
        _running_asr_instances.clear()

class ASR:
    def __init__(self, lang, model_path, nlp=None, translator=None):
        print(f"[DEBUG] ASR __init__ called for lang={lang}, model_path={model_path}")
        try:
            print(f"[DEBUG] ASR __init__: Attempting to load Vosk Model from path: {model_path}")
            self.model = Model(model_path)
            print(f"[DEBUG] ASR __init__: Successfully loaded Vosk Model for path: {model_path}")
            # Signal that this model has just finished loading
            from asr import signal_model_loaded
            signal_model_loaded(lang)
        except Exception as e:
            print(f"[ERROR] ASR __init__: Failed to load Vosk Model from path: {model_path}. Error: {e}")
            raise RuntimeError(f"Failed to load Vosk model for language {lang} from path {model_path}") from e

        self.recognizer = KaldiRecognizer(self.model, 16000)
        print(f"[DEBUG] KaldiRecognizer created for model: {model_path}")
        self.lang = lang
        self._terminated = False
        self.translator = translator
        self.nlp = nlp
        register_asr_instance(self)
        print(f"[DEBUG] ASR instance registered for lang={lang}")

    def recognize_stream(self):
        """Generator that yields recognition results. First yield is a ready message."""
        # This is sent *after* __init__ is complete, so model is loaded.
        yield {'status': 'ready_and_listening', 'message': f'Model {self.lang.upper()} is loaded and listening.'}

        p = pyaudio.PyAudio()
        stream = p.open(rate=16000, channels=1, format=pyaudio.paInt16, 
                        input=True, output=False, frames_per_buffer=2048)
        stream.start_stream()
        print(f"[DEBUG] PyAudio stream started for lang={self.lang}")

        try:
            while not self._terminated:
                data = stream.read(2048, exception_on_overflow=False)
                if self._terminated or len(data) == 0:
                    break
                
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                reduced_chunk = nr.reduce_noise(y=audio_chunk, sr=16000, stationary=True, prop_decrease=1.0)
                data = reduced_chunk.tobytes()

                if self.recognizer.AcceptWaveform(data):
                    res = json.loads(self.recognizer.Result())
                    text = res.get("text", "")
                    if text:
                        print(f"[DEBUG] ASR result for {self.lang}: {text}")
                        result_obj = {'transcription': text}
                        if self.lang != 'en':
                            translated = translate_to_english(text, self.lang)
                            print(f"[DEBUG] Translated result for {self.lang}: {translated}")
                            result_obj['translation'] = translated
                        else:
                            result_obj['translation'] = text
                        yield result_obj
        finally:
            print(f"[DEBUG] ASR recognize_stream finally block for lang={self.lang}")
            stream.stop_stream()
            stream.close()
            p.terminate()
            # Cleanup ASR resources
            unregister_asr_instance(self)
            print(f"[DEBUG] ASR instance unregistered for lang={self.lang}")
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'recognizer'):
                del self.recognizer
            print(f"[DEBUG] self.model and self.recognizer deleted for lang={self.lang}")

    def listen(self):
        print(f"[DEBUG] ASR listen called for lang={self.lang}")
        p = pyaudio.PyAudio()
        stream = p.open(rate=16000, channels=1, format=pyaudio.paInt16, 
                        input=True, output=False, frames_per_buffer=2048)
        stream.start_stream()
        final_text = ""
        translated = ""
        try:
            while True:
                if self._terminated:
                    print(f"[DEBUG] ASR listen loop terminated for lang={self.lang}")
                    break
                data = stream.read(2048, exception_on_overflow=False)
                if self._terminated or len(data) == 0:
                    print(f"[DEBUG] ASR listen loop terminated (no data or terminated) for lang={self.lang}")
                    break
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                reduced_chunk = nr.reduce_noise(y=audio_chunk, sr=16000, stationary=True, prop_decrease=1.0)
                data = reduced_chunk.tobytes()
                if self.recognizer.AcceptWaveform(data):
                    res = json.loads(self.recognizer.Result())
                    final_text = res["text"]
                    print(f"[DEBUG] ASR result for {self.lang}: {final_text}")
                    if self.lang != 'en' and final_text:
                        translated = translate_to_english(final_text, self.lang)
                        print(f"[DEBUG] Translated result for {self.lang}: {translated}")
                    else:
                        translated = final_text
                # No partial result logic here
        finally:
            print(f"[DEBUG] ASR listen finally block for lang={self.lang}")
            stream.stop_stream()
            stream.close()
            p.terminate()
            unregister_asr_instance(self)
            print(f"[DEBUG] ASR instance unregistered for lang={self.lang}")
            # Restore del self.model and del self.recognizer
            del self.model
            del self.recognizer
            print(f"[DEBUG] self.model and self.recognizer deleted for lang={self.lang}")

    def terminate(self):
        print(f"[DEBUG] ASR terminate called for lang={self.lang}")
        self._terminated = True
