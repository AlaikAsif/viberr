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
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.lang = lang
        self._terminated = False
        self.translator = translator
        self.nlp = nlp
        register_asr_instance(self)

    def listen(self):
        p = pyaudio.PyAudio()
        stream = p.open(rate=16000, channels=1, format=pyaudio.paInt16, 
                        input=True, output=False, frames_per_buffer=2048)
        stream.start_stream()
        final_text = ""
        translated = ""
        try:
            while True:
                if self._terminated:
                    break
                data = stream.read(2048, exception_on_overflow=False)
                if self._terminated or len(data) == 0:
                    break
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                reduced_chunk = nr.reduce_noise(y=audio_chunk, sr=16000, stationary=True, prop_decrease=1.0)
                data = reduced_chunk.tobytes()
                if self.recognizer.AcceptWaveform(data):
                    res = json.loads(self.recognizer.Result())
                    final_text = res["text"]
                    if self.lang != 'en' and final_text:
                        translated = translate_to_english(final_text, self.lang)
                    else:
                        translated = final_text
                # No partial result logic here
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            unregister_asr_instance(self)
            del self.model
            del self.recognizer
        return final_text, translated

    def terminate(self):
        self._terminated = True

def asr_stream_listen(lang, model_path, translator=None):
    asr = ASR(lang, model_path, translator=translator)
    p = pyaudio.PyAudio()
    stream = p.open(rate=16000, channels=1, format=pyaudio.paInt16, 
                    input=True, output=False, frames_per_buffer=2048)
    stream.start_stream()
    try:
        while True:
            if asr._terminated:
                break
            data = stream.read(2048, exception_on_overflow=False)
            if asr._terminated or len(data) == 0:
                break
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            reduced_chunk = nr.reduce_noise(y=audio_chunk, sr=16000, stationary=True, prop_decrease=1.0)
            data = reduced_chunk.tobytes()
            if asr.recognizer.AcceptWaveform(data):
                res = json.loads(asr.recognizer.Result())
                text = res.get("text", "")
                if text:
                    if lang == 'en':
                        yield json.dumps({"text": text, "lang": lang})
                    else:
                        translated = translate_to_english(text, lang)
                        yield json.dumps({"text": text, "lang": lang, "english": translated})
                if text.lower() == "stop listening":
                    break
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        unregister_asr_instance(asr)
        del asr.model
        del asr.recognizer
