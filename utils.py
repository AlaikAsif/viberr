import json

def load_json(file_path):
    """Load a JSON file and return its content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"file {file_path} is not a valid JSON file.")
