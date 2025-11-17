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
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from src.utils.utils import clean_sentence, get_synonyms, ensure_nltk_data
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
                tokens = nltk.word_tokenize(pattern.lower())
                tokens = [t for t in tokens if t not in self.ignore_words]
                w = [self.lemmatizer.lemmatize(t) for t in tokens]
                
                if not w:
                    continue
                    
                self.words.extend(w)
                self.documents.append((w, intent['tag']))
                
                for word in tokens: 
                    for syn in get_synonyms(word):
                        syn_tokens = nltk.word_tokenize(syn.lower())
                        syn_clean = [s for s in syn_tokens if s not in self.ignore_words]
                        syn_lem = [self.lemmatizer.lemmatize(s) for s in syn_clean]
                        if syn_lem:
                            self.words.extend(syn_lem)
                            self.documents.append((syn_lem, intent['tag']))
                            
            if intent['tag'] not in self.classes:
                self.classes.append(intent['tag'])

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
        
        train_data, test_data = train_test_split(
            training, test_size=0.2, random_state=42, stratify=[t[1] for t in training]
        )
        
        X_train = np.array([t[0] for t in train_data])
        y_train = np.array([t[1] for t in train_data])
        X_test = np.array([t[0] for t in test_data])
        y_test = np.array([t[1] for t in test_data])
        
        return X_train, X_test, y_train, y_test

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
        ensure_nltk_data()
        self.load_data()
        self.preprocess()
        X_train, X_test, y_train, y_test = self.create_training_data()
        
        model = self.build_model(len(X_train[0]), len(y_train[0]))
        
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        print(f"Entrenando con {len(X_train):,} muestras (train) y {len(X_test):,} muestras (test)")
        
        history = model.fit(
            X_train, y_train,
            epochs=300,
            batch_size=8,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        best_epoch = np.argmax(history.history['val_accuracy'])
        best_val_acc = history.history['val_accuracy'][best_epoch]
        best_val_loss = history.history['val_loss'][best_epoch]
        final_epoch = len(history.history['accuracy'])
        
        print(f"\nBEST RESULT (epoch {best_epoch + 1}):")
        print(f"   → Test Accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        print(f"   → Test Loss     : {best_val_loss:.4f}")
        print(f"   → It stopped at epoch {final_epoch} (early stopping)")
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save(MODEL_PATH)
        with open(WORDS_PATH, 'wb') as f: pickle.dump(self.words, f)
        with open(CLASSES_PATH, 'wb') as f: pickle.dump(self.classes, f)
        
        metrics = {
            "best_epoch": int(best_epoch + 1),
            "final_epoch": int(final_epoch),
            "best_val_accuracy": float(best_val_acc),
            "best_val_loss": float(best_val_loss),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "total_patterns": len(self.intents["intents"]),
            "total_intents": len(self.classes),
            "notes": "BoW + MLP + synonym augmentation + early stopping"
        }
        
        metrics_path = os.path.join(MODEL_DIR, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print("Model and metrics saved successfully.")