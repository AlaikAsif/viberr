from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
from asr import multi_asr_listen, LANG_MODELS, cleanup_unused_models, ensure_model_downloaded, get_model_progress, is_model_ready, get_model_status_info
from recognizer import asr_stream_listen
import os
import time
import threading

app = Flask(__name__)

# Dictionary to hold background download threads
_download_threads = {}
_thread_lock = threading.Lock()

@app.route('/')
def index():
    lang = request.args.get('lang') or 'en'
    if lang not in LANG_MODELS:
        lang = 'en'
    models_in_use = lang
    print(f"[DEBUG] / route called with lang={lang}")
    # Pass model readiness to template for JS polling
    return render_template('index.html', lang=lang, models_in_use=models_in_use)

@app.route('/listen', methods=['POST'])
def listen_route():
    lang = request.form.get('lang') or 'en'
    if lang not in LANG_MODELS:
        return jsonify({'status': 'error', 'message': 'Invalid language selected.'}), 400

    print(f"[DEBUG] /listen (POST) route called for lang={lang}")

    status = get_model_status_info(lang)

    if status['status'] in ['downloading', 'extracting']:
        print(f"[DEBUG] /listen POST - Download/extraction already in progress for {lang}")
        return jsonify(status)

    if status['status'] == 'ready':
        print(f"[DEBUG] /listen POST - Model for {lang} is ready.")
        # The frontend should ideally initiate streaming directly.
        # This response is a fallback.
        return jsonify(status)

    # If model is not ready and not being downloaded, start download in a background thread.
    with _thread_lock:
        # Double-check inside the lock to prevent race conditions
        if lang not in _download_threads or not _download_threads[lang].is_alive():
            print(f"[DEBUG] /listen POST - Starting background download thread for {lang}")
            thread = threading.Thread(target=ensure_model_downloaded, args=(lang,))
            _download_threads[lang] = thread
            thread.start()
            # Return a status indicating the download has begun
            return jsonify({'status': 'download_started', 'message': f'Download started for {lang} model.'})
        else:
            # A thread was started between the outer check and acquiring the lock
            print(f"[DEBUG] /listen POST - Download/extraction was just started for {lang}")
            return jsonify(get_model_status_info(lang))

@app.route('/stream')
def stream_route():
    lang = request.args.get('lang') or 'en'
    if lang not in LANG_MODELS:
        lang = 'en'
    print(f"[DEBUG] /stream route called with lang={lang}")
    model_path = LANG_MODELS.get(lang)
    print(f"[DEBUG] /stream route - model_path resolved: {model_path}")
    # Check if model is ready
    if lang != 'en' and not is_model_ready(lang):
        print(f"[DEBUG] /stream route - Model for {lang} not ready, redirecting to wait page")
        ensure_model_downloaded(lang)
        next_url = url_for('index', lang=lang)
        return render_template('wait.html', next_url=next_url, models_in_use=lang)
    # Clean up unused models before running recognition
    cleanup_unused_models(lang)
    print(f"[DEBUG] /stream route - Calling asr_stream_listen with lang={lang} and model_path={model_path}")
    def event_stream():
        for result in asr_stream_listen(lang, model_path):
            yield f'data: {result}\n\n'
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/recognize', methods=['POST'])
def recognize_route():
    lang = request.form.get('lang') or 'en'
    models_in_use = lang
    if not is_model_ready(lang):
        return "Model not ready. Please go back and wait for the download to complete.", 400

    print(f"[DEBUG] /recognize route POST - Model ready for {lang}, starting recognition")
    cleanup_unused_models(lang)
    result = multi_asr_listen([lang])
    
    if lang != 'en':
        english_result = result['all_results'][lang][1]
        return f'<h2>Model: {models_in_use}</h2>' \
               f'<h2>English Translation:</h2><p>{english_result}</p>'
    else:
        english_result = result['all_results'][lang][0]
        return f'<h2>Model: {models_in_use}</h2>' \
               f'<h2>Transcription:</h2><p>{english_result}</p>'

@app.route('/model_progress')
def model_progress_route():
    lang = request.args.get('lang') or 'en'
    progress = get_model_progress(lang)
    return jsonify(progress)

@app.route('/terminate', methods=['POST'])
def terminate_instance():
    print('[DEBUG] /terminate route called - cleaning up all models and resources')
    # Clean up all models/resources (pass None or suitable arg to cleanup_unused_models)
    cleanup_unused_models(None)
    # Optionally, stop any ongoing downloads or loading tasks here if implemented
    # Optionally, set a global flag or state to indicate termination if needed
    return jsonify({'status': 'terminated', 'message': 'Instance terminated and loading stopped.'})

@app.route('/model_status')
def model_status_route():
    lang = request.args.get('lang') or 'en'
    # Only check status, do NOT trigger downloads here
    # Downloads should only be triggered by /listen route when user clicks "Start Listening"
    status = get_model_status_info(lang)
    return jsonify(status)

if __name__ == '__main__':
    app.run(debug=True)
