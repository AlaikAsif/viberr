from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
from asr import multi_asr_listen, LANG_MODELS, cleanup_unused_models, ensure_model_downloaded, get_model_progress, is_model_ready, get_model_status_info
from recognizer import ASR
import os
import time
import threading
import json  # for SSE messages

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
def stream():
    lang = request.args.get('lang', 'en')
    if lang not in LANG_MODELS:
        return Response(status=400)

    model_path = LANG_MODELS[lang]

    if not is_model_ready(lang):
        def error_generate():
            yield f"data: {json.dumps({'status': 'error', 'message': 'Model not ready on disk.'})}\n\n"
        return Response(error_generate(), mimetype='text/event-stream')

    def generate():
        try:
            asr = ASR(lang, model_path)
            for result in asr.recognize_stream():
                yield f"data: {json.dumps(result)}\n\n"
        except Exception as e:
            print(f"[ERROR] /stream: Failed to initialize or run ASR for lang={lang}. Error: {e}")
            yield f"data: {json.dumps({'status': 'error', 'message': 'Failed to start recognition stream.'})}\n\n"
        finally:
            print(f"[DEBUG] /stream generate() for {lang} finished.")

    return Response(generate(), mimetype='text/event-stream')

@app.route('/recognize', methods=['POST'])
def recognize_route():
    lang = request.form.get('lang') or 'en'
    models_in_use = lang
    if not is_model_ready(lang):
        return "Model not ready. Please go back and wait for the download to complete.", 400

    print(f"[DEBUG] /recognize route POST - Model ready for {lang}, starting recognition")
    cleanup_unused_models(lang)
    result = multi_asr_listen([lang])
    # Get both original and translated results
    orig, trans = result['all_results'][lang]
    # Return both transcription and translation as JSON object
    return jsonify({
        'model': models_in_use,
        'language': lang,
        'transcription': orig,
        'translation': trans
    })

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
