import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "intents.json")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "chatbot_model.keras")
WORDS_PATH = os.path.join(MODEL_DIR, "words.pkl")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.pkl")