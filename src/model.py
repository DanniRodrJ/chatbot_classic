# src/model.py
import nltk
import json
import pickle
import numpy as np
import os
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from src.utils.utils import clean_sentence, get_synonyms
from src.config import MODEL_DIR, DATA_PATH, MODEL_PATH, WORDS_PATH, CLASSES_PATH

class ChatbotTrainer:
    def __init__(self):
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = set(".,!?;:'\"()[]{}|_/@#$%^&*~`+=-<>")

    def load_data(self):
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            self.intents = json.load(f)

    def preprocess(self):
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)
                self.documents.append((w, intent['tag']))
                for word in w:
                    for syn in get_synonyms(word):
                        self.words.extend(nltk.word_tokenize(syn))
                        self.documents.append((nltk.word_tokenize(syn), intent['tag']))
            if intent['tag'] not in self.classes:
                self.classes.append(intent['tag'])

        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))

    def create_training_data(self):
        training = []
        output_empty = [0] * len(self.classes)
        for doc in self.documents:
            bag = [1 if w in [self.lemmatizer.lemmatize(w.lower()) for w in doc[0]] else 0 for w in self.words]
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])
        random.shuffle(training)
        return np.array([t[0] for t in training]), np.array([t[1] for t in training])

    def build_model(self, input_size, output_size):
        model = Sequential([
            Input(shape=(input_size,)),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(output_size, activation='softmax')
        ])
        lr = ExponentialDecay(0.01, decay_steps=10000, decay_rate=0.9)
        sgd = SGD(learning_rate=lr, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model

    def train(self):
        self.load_data()
        self.preprocess()
        X, y = self.create_training_data()
        model = self.build_model(len(X[0]), len(y[0]))
        model.fit(X, y, epochs=200, batch_size=5, verbose=1)
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save(MODEL_PATH)
        with open(WORDS_PATH, 'wb') as f: pickle.dump(self.words, f)
        with open(CLASSES_PATH, 'wb') as f: pickle.dump(self.classes, f)
        print("Modelo entrenado y guardado.")