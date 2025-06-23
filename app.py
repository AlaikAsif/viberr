from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
from asr import multi_asr_listen, LANG_MODELS, cleanup_unused_models, ensure_model_downloaded, get_model_progress, is_model_ready
from recognizer import asr_stream_listen
import os
import time

app = Flask(__name__)

@app.route('/')
def index():
    lang = request.args.get('lang') or 'en'
    if lang not in LANG_MODELS:
        lang = 'en'
    models_in_use = lang
    # Pass model readiness to template for JS polling
    return render_template('index.html', lang=lang, models_in_use=models_in_use)

@app.route('/listen', methods=['GET'])
def listen_route():
    lang = request.args.get('lang') or 'en'
    if lang not in LANG_MODELS:
        lang = 'en'
    models_in_use = lang
    # Check if model is missing (robust)
    missing = lang != 'en' and not is_model_ready(lang)
    # AJAX polling for readiness
    if request.args.get('ajax') == '1':
        if not missing:
            return jsonify({'ready': True, 'models_in_use': models_in_use})
        else:
            return jsonify({'ready': False, 'models_in_use': models_in_use})
    if missing:
        ensure_model_downloaded(lang)
        next_url = url_for('index', lang=lang)
        return render_template('wait.html', next_url=next_url, models_in_use=models_in_use)
    # Clean up unused models before running recognition
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

@app.route('/stream')
def stream_route():
    lang = request.args.get('lang') or 'en'
    if lang not in LANG_MODELS:
        lang = 'en'
    # Clean up unused models before running recognition
    cleanup_unused_models(lang)
    def event_stream():
        for result in asr_stream_listen(lang, LANG_MODELS[lang]):
            yield f'data: {result}\n\n'
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/model_progress')
def model_progress_route():
    lang = request.args.get('lang') or 'en'
    progress = get_model_progress(lang)
    return jsonify(progress)

if __name__ == '__main__':
    app.run(debug=True)
