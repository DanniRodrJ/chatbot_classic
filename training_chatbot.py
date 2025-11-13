import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import random


data_file = open("documentacion.json", "r", encoding="utf-8").read()
intents = json.loads(data_file)

lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = set(".,!?;:'\"()[]{}|_/@#$%^&*~`+=-<>")


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())  # Agrega el sinónimo
    return list(synonyms)

# Recorre cada intención y sus patrones en el archivo JSON
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokeniza las palabras en cada patrón y las agrega a la lista de palabras
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Agrega el par (patrón, etiqueta) a la lista de documentos
        documents.append((w, intent['tag']))
        
        
        # Genera sinónimos para cada palabra en el patrón
        for word in w:
            synonyms = get_synonyms(word)
            for synonym in synonyms:
                words.extend(nltk.word_tokenize(synonym))
                documents.append((nltk.word_tokenize(synonym), intent['tag']))
        
        
        # Si la etiqueta no está en la lista de clases, la agrega
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
# Lemantiza las palabras y las convierte en minisculas, excluyendo las palabras ignoradas
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Guarda las listas de palabras y clases en archivos pickle
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

# Crear datos de entrenamiento (Bag Words)
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for word in words:
        # Crea una bolsa de palabras binaria para cada patrón
        bag.append(1) if word in pattern_words else bag.append(0)
    output_row = list(output_empty)
    # Crea un vector de salida con un 1 en la posicion correspondiente de la intencion
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
    
# Mezcla aleatoriamente el conjunto de entrenamiento 
random.shuffle(training)

# Divide el conjunto de entrenamiento en caracteristicas (train_x) y etiquetas (train_y)
train_x = [row[0] for row in training]
train_y = [row[1] for row in training]

train_x = np.array(train_x)
train_y = np.array(train_y)

# Crea el modelo de red neuronal
model = Sequential([
    Input(shape=(len(train_x[0]),)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

# Configura el optimizador con una tasa de aprendizaje exponencialmente decreciente
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9
)
sgd = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrena el modelo con el conjunto de entrenamiento
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Guarda el modelo entrenado
model.save('chatbot_model.keras')
print('Model created')