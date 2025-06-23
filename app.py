<<<<<<< HEAD
from flask import Flask, render_template, request, redirect, url_for, jsonify
from asr import multi_asr_listen, LANG_MODELS
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/listen', methods=['GET'])
def listen_route():
    langs = request.args.getlist('lang')
    if not langs:
        langs = ['en']
    # Check if any model is missing
    missing = [lang for lang in langs if lang != 'en' and not os.path.exists(LANG_MODELS[lang])]
    models_in_use = ', '.join(langs)
    if missing and not request.args.get('waited'):
        # Show wait page and redirect back to /listen with waited=1
        next_url = url_for('listen_route', **{**request.args, 'waited': 1})
        return render_template('wait.html', next_url=next_url, models_in_use=models_in_use)
    result = multi_asr_listen(langs)
    # If the request is from an API client (Accept: application/json), return JSON
    if request.headers.get('Accept') == 'application/json':
        return jsonify(result)
    # Otherwise, return HTML for browser
    return f'<h2>Models in use: {models_in_use}</h2>' \
           f'<h2>Most Relevant (English):</h2><p>{result["best_english"]}</p>' \
           f'<h3>All Results:</h3>' + ''.join([f'<p><b>{lang}:</b> {text} <br> <b>EN:</b> {eng}</p>' for lang, (text, eng) in result["all_results"].items()])

if __name__ == '__main__':
    app.run(debug=True)
=======
# Entry point for the Viberr Trainer Web app
# (Flask or similar Python web framework can be used)

if __name__ == "__main__":
    print("Viberr Trainer Web app starting...")
>>>>>>> 3f2c0e0b96799d4a834c4c8a9edefa20cbdc9252
