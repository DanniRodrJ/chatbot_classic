# src/chatbot.py
import pickle
import json
import numpy as np
import re
import random
from tensorflow.keras.models import load_model
from src.utils.utils import clean_sentence, get_synonyms, extract_order_number
from src.config import DATA_PATH, MODEL_PATH, WORDS_PATH, CLASSES_PATH

class Chatbot:
    def __init__(self):
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            self.intents = json.load(f)
        self.words = pickle.load(open(WORDS_PATH, 'rb'))
        self.classes = pickle.load(open(CLASSES_PATH, 'rb'))
        self.model = load_model(MODEL_PATH)

    def predict_intent(self, sentence: str, last_intent: str = None):
        bow = self._bag_of_words(sentence)
        res = self.model.predict(np.array([bow]), verbose=0)[0]
        ERROR_THRESHOLD = 0.5
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)

        if not results:
            for intent in self.intents['intents']:
                for pattern in intent['patterns']:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        return [{'intent': intent['tag'], 'probability': '1.0'}]
            return [{'intent': last_intent or '', 'probability': '0.6'}]

        return [{'intent': self.classes[r[0]], 'probability': str(r[1])} for r in results]

    def get_response(self, intents_list, last_response=None):
        if not intents_list:
            return "I'm sorry, I didn't understand. Could you rephrase?"
        tag = intents_list[0]['intent']
        for intent in self.intents['intents']:
            if intent['tag'] == tag:
                responses = intent['responses']
                if last_response and len(responses) > 1:
                    filtered = [r for r in responses if r != last_response]
                    return random.choice(filtered or responses)
                return random.choice(responses)
        return "I don't have a response for that."

    def _bag_of_words(self, sentence):
        sentence_words = clean_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w or word in get_synonyms(w):
                    bag[i] = 1
        return np.array(bag)