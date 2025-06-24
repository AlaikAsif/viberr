import json
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
from translate_util import translate_to_english
from recognizer import ASR, register_asr_instance, unregister_asr_instance, terminate_all_asr_instances

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

# Progress tracking for model downloads
_model_progress = {}
# Track recently loaded models for showing alerts
_recently_loaded_models = set()
_recently_loaded_lock = threading.Lock()

def signal_model_loaded(lang):
    """Signal that a model has just finished loading into memory."""
    with _recently_loaded_lock:
        _recently_loaded_models.add(lang)
        print(f"[DEBUG] Model {lang} signaled as just loaded")

def check_and_clear_model_loaded(lang):
    """Check if a model was recently loaded and clear the flag."""
    with _recently_loaded_lock:
        if lang in _recently_loaded_models:
            _recently_loaded_models.remove(lang)
            print(f"[DEBUG] Model {lang} just-loaded flag cleared")
            return True
        return False

def get_model_progress(lang):
    return _model_progress.get(lang, {"status": "idle", "progress": 0})

def safe_remove(path, retries=5, delay=0.2):
    import time
    for i in range(retries):
        try:
            if os.path.exists(path):
                os.remove(path)
            return
        except PermissionError:
            if i == retries - 1:
                raise
            time.sleep(delay)

def ensure_model_downloaded(lang):
    if lang == 'en':
        return  # English model is always present
    model_path = LANG_MODELS[lang]
    zip_path = f"{model_path}.zip"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    def download_zip():
        url = MODEL_URLS[lang]
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            downloaded = 0
            _model_progress[lang] = {"status": "downloading", "progress": 0}
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = int(downloaded * 100 / total) if total else 0
                        _model_progress[lang] = {"status": "downloading", "progress": percent}
            _model_progress[lang] = {"status": "downloaded", "progress": 100}
    extracted = False
    if not os.path.exists(model_path) and os.path.exists(zip_path):
        try:
            _model_progress[lang] = {"status": "extracting", "progress": 0}
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("models")
            _model_progress[lang] = {"status": "ready", "progress": 100}
            extracted = True
        except Exception as e:
            print(f"Error extracting model zip for '{lang}': {e}. Retrying download...")
            try:
                safe_remove(zip_path)
            except Exception as e_rm:
                print(f"Failed to remove zip after extraction error: {e_rm}")
            download_zip()
            try:
                _model_progress[lang] = {"status": "extracting", "progress": 0}
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall("models")
                _model_progress[lang] = {"status": "ready", "progress": 100}
                extracted = True
            except Exception as e2:
                print(f"Failed again to extract model zip for '{lang}': {e2}")
            finally:
                try:
                    safe_remove(zip_path)
                except Exception as e_rm:
                    print(f"Failed to remove zip after second extraction error: {e_rm}")
        print('models directory after extraction:', os.listdir('models'))
    if not os.path.exists(model_path) and not extracted:
        print(f"Model for '{lang}' not found. Downloading...")
        download_zip()
        try:
            _model_progress[lang] = {"status": "extracting", "progress": 0}
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("models")
            _model_progress[lang] = {"status": "ready", "progress": 100}
        except Exception as e:
            print(f"Error extracting downloaded model zip for '{lang}': {e}")
        finally:
            try:
                safe_remove(zip_path)
            except Exception as e_rm:
                print(f"Failed to remove zip after download extraction error: {e_rm}")
        print('models directory after extraction:', os.listdir('models'))
    else:
        if os.path.exists(zip_path):
            try:
                safe_remove(zip_path)
            except Exception as e_rm:
                print(f"Failed to remove zip after model already exists: {e_rm}")
        _model_progress[lang] = {"status": "ready", "progress": 100}
        print(f"Model for '{lang}' already exists.")

def cleanup_unused_models(current_lang):
    """
    Terminate all running ASR instances before starting a new one.
    """
    print(f"[DEBUG] cleanup_unused_models called for current_lang={current_lang}")
    terminate_all_asr_instances()
    print(f"[DEBUG] All ASR instances terminated.")

def multi_asr_listen(langs):
    print(f"[DEBUG] multi_asr_listen called with langs={langs}")
    results = {}
    threads = []
    def run_asr(lang):
        current_model_path = LANG_MODELS.get(lang)
        print(f"[DEBUG] run_asr thread for lang={lang}: preparing to call ASR constructor with model_path={current_model_path}")
        # Use recognizer.ASR and pass model path
        asr = ASR(lang, current_model_path)
        original, translated = asr.listen()
        print(f"[DEBUG] ASR for {lang}: original='{original}', translated='{translated}'")
        results[lang] = (original, translated)
    for lang in langs:
        t = threading.Thread(target=run_asr, args=(lang,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    # Always select the longest non-empty English translation (translated or original)
    best_english = ""
    for lang, (orig, trans) in results.items():
        candidate = trans.strip() if trans and trans.strip() else orig.strip()
        if len(candidate) > len(best_english):
            best_english = candidate
    print(f"[DEBUG] multi_asr_listen finished, best_english='{best_english}'")
    return {"all_results": results, "best_english": best_english}

def is_model_ready(lang):
    """
    Check if the Vosk model for the given language is truly ready (key file exists).
    For most Vosk models, 'am/final.mdl' is a reliable indicator.
    Also check for 'model.conf' in both root and conf/ subdir.
    """
    if lang == 'en':
        return os.path.exists(LANG_MODELS['en'])
    model_path = LANG_MODELS.get(lang)
    if not model_path or not os.path.isdir(model_path):
        # Only log when model is actually missing (less noise for radio button clicks)
        print(f"[DEBUG] is_model_ready for {lang}: model_path {model_path} not found or not a directory")
        return False
    key_file_1 = os.path.join(model_path, 'am', 'final.mdl')
    key_file_2 = os.path.join(model_path, 'model.conf')
    key_file_3 = os.path.join(model_path, 'conf', 'model.conf')
    ready = os.path.exists(key_file_1) or os.path.exists(key_file_2) or os.path.exists(key_file_3)
    # Only log detailed info when something interesting happens or model is not ready
    if not ready:
        print(f"[DEBUG] is_model_ready for {lang}: key_file_1={key_file_1} ({os.path.exists(key_file_1)}), key_file_2={key_file_2} ({os.path.exists(key_file_2)}), key_file_3={key_file_3} ({os.path.exists(key_file_3)}), ready={ready}")
    return ready

def get_model_status_info(lang):
    """
    Get comprehensive model status information without triggering downloads.
    Returns status dict with status, progress, and message.
    """
    if lang == 'en':
        return {
            'status': 'ready',
            'progress': 100,
            'message': 'English model is always ready.'
        }
    
    if is_model_ready(lang):
        # Check if this model was just loaded
        just_loaded = check_and_clear_model_loaded(lang)
        return {
            'status': 'ready',
            'progress': 100,
            'message': 'Model loaded and ready.',
            'just_loaded': just_loaded
        }
    
    # Get current progress if not ready
    progress_data = get_model_progress(lang)
    # Treat any non-active states as not_downloaded so download button shows
    raw_status = progress_data.get('status', 'not_downloaded')
    status = raw_status if raw_status in ['downloading', 'extracting'] else 'not_downloaded'
    progress = progress_data.get('progress', 0)
    
    if status == 'downloading':
        message = f"Downloading model... {progress}%"
    elif status == 'extracting':
        message = f"Extracting model... {progress}%"
    elif status == 'downloaded':
        message = 'Model downloaded, preparing...'
    else:
        message = 'Model not downloaded.'
    
    return {
        'status': status,
        'progress': progress,
        'message': message
    }
