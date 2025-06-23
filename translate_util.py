from deep_translator import GoogleTranslator

def translate_to_english(text, src_lang):
    if not text or not text.strip():
        return text
    # Normalize language code (e.g., 'en-US' -> 'en')
    src_lang = src_lang.split('-')[0].lower() if src_lang else 'auto'
    if src_lang == 'en':
        return text
    try:
        translated = GoogleTranslator(source=src_lang, target='en').translate(text)
        return translated
    except Exception as e:
        print(f"Translation error (src_lang={src_lang}): {e}")
        return text  # Fallback to original text if translation fails
