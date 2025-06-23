import sys
from asr import ensure_model_downloaded, LANG_MODELS

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_model.py <lang_code>")
        print("Available languages:", ', '.join(LANG_MODELS.keys()))
        sys.exit(1)
    lang = sys.argv[1]
    if lang not in LANG_MODELS:
        print(f"Language '{lang}' is not supported.")
        print("Available languages:", ', '.join(LANG_MODELS.keys()))
        sys.exit(1)
    ensure_model_downloaded(lang)
    print(f"Model for '{lang}' is ready.")
