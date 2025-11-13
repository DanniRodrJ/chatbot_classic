import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from nltk.corpus import wordnet
import re

lemmantizer = WordNetLemmatizer() 
intents = json.loads(open('documentacion.json', 'r', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')

# Create a function to clean up the user's input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmantizer.lemmatize(word.lower()) for word in sentence_words] 

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())  # Agrega el sinónimo
    return list(synonyms)


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w or word in get_synonyms(w):
                bag[i] = 1
            
    return np.array(bag)
    
def predict_class(sentence, last_intent=None):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.5
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Si no hay coincidencias directas, intenta con regex
    if not results:
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                if re.search(pattern, sentence, re.IGNORECASE):
                    return [{'intent': intent['tag'], 'probability': '1.0'}]
        if last_intent:
            return [{'intent': last_intent, 'probability': '0.6'}]
        return [{'intent': 'no_entendido', 'probability': '1.0'}]
    
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json, last_response=None):
    if not intents_list:
        return "Lo siento, no entendí. ¿Puedes reformularlo?"
    
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            responses = intent['responses']
            # Evitar repetir la misma respuesta
            if last_response and len(responses) > 1:
                filtered = [r for r in responses if r != last_response]
                return random.choice(filtered) if filtered else random.choice(responses)
            return random.choice(responses)
    return "No tengo respuesta para eso."

