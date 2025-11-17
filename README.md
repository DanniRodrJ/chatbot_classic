# Classic Intent Classifier: BoW + MLP (Non-LLM NLU)

![image](/assets/image.png)

A fully controllable, explainable intent classification system using **Bag of Words (BoW)** and a **Multilayer Perceptron (MLP)**.This system is optimized for customer service centers with a fixed set of intents, providing a **low-cost**, highly transparent solution.

## 🚀 ```Demo```

![Demo](/assets/demo.gif)

## ⚙️ ```Tech Stack```

| Technology | Purpose |
|----------|--------|
| **Python** | Core language |
| **TensorFlow / Keras** | MLP model training & inference |
| **NLTK** | Tokenization, lemmatization, synonym expansion |
| **Streamlit** | Production-ready local UI |
| **BoW** | Text-to-vector encoding |

## 🔬 ```Technical Overview```

| Component | Implementation |
|---------|----------------|
| **Model** | MLP (128 → 64 → N) with Dropout |
| **Feature Engineering** | BoW + lemmatization + WordNet synonyms |
| **Preprocessing** | NLTK (tokenization, lemmatization) |
| **Context** | Previous intent state (`last_intent`) |
| **Entity Extraction** | **Regex + Slot Filling** |
| **Framework** | TensorFlow/Keras |
| **Interface** | Streamlit (local deployment) |
| **Architecture** | Modular (`src/`, `data/`, `models/`) |

## 📁 ```Project Structure```

```bash
chatbot_classic/
├── src/
│   ├── __init__.py
│   ├── config.py         # Path management
│   ├── model.py          # Training pipeline
│   ├── chatbot.py        # Inference engine
│   └── utils.py          # Preprocessing + NLTK auto-download
│
├── data/
│   └── documentacion.json
│
├── models/               # Generated artifacts
│   ├── .gitkeep
│   ├── chatbot_model.keras
│   ├── words.pkl
│   └── classes.pkl
│
├── app.py                # Streamlit frontend
├── training_chatbot.py   # Training entrypoint
├── requirements.txt
├── .gitignore
└── README.md
```

## ⬇️ ```Inference Pipeline```

The inference engine processes the input to determine intent and fill in key slots, such as order numbers.

```bash
Input: "I want to return a damaged item"
        ↓
Tokenization → Lemmatization → Synonym Expansion
        ↓
Bag of Words (compare with original words and expanded synonyms) → [0, 1, 0, 1, ...] (binary vector)
        ↓
MLP → [0.02, 0.91, 0.01, ...] → intent: "return"
        ↓
Context & Entity Check: last_intent = "return"
- last_intent = "return"
- Regex: extract the order number if present (Slot Filling)
        ↓
Output: "Sorry to hear that. Can you provide the order number?"
```

## 🧠 ```Neural Network Architecture```

```python
Sequential([
    Input(shape=(vocab_size,)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])
```

- **Optimizer**: SGD with exponential decay
- **Loss**: categorical_crossentropy
- **Training**: 200 epochs, batch size 5

## ⭐ ```Key Features```

|Feature|Description|
|-------|-----------|
|**Intent Classification**|11 predefined intents|
|**Simple Context** | Uses last_intent for conversational flow|
|**Slot Filling (Enhanced)**|Order numbers extracted via regex and used to format the response|
|**Response Deduplication**|Filters repeated responses|
|**Synonym Augmentation**|Automatic pattern expansion via WordNet|
|**Robust Fallback**|Regex + previous intent if confidence < 0.5|

## 📈 ```Training Metrics```
The model is trained for 200 epochs. The following is the accuracy achieved at the end of the process, demonstrating the model's ability to learn from intention patterns.

|Metrics|Final Value ( Epoch 200)|
|-------|------|
|**Accuracy**|~0.74 (74%)|
|**Loss**|~0.59|

## ⚠️ ```Known Limitations```

| Limitation                  | Mitigation                            | Impact |
|----------------------------|----------------------------------------|--------|
| **No sequence modeling**     | Accepted (BoW design)                  | "I don't want" may be treated like "I want" |
| **Static vocabulary**    | Retrain to expand               | No runtime adaptation to new terms |
| **No long-term memory** | `last_intent as minimal state     | Flows >2 steps lose context |
| **Limited entity support**     | Regex + manual patterns               | Only detects predefined formats |

> **These are inherent to the classic design and fully controlled**

## ⏭️ ```Project Evolution```

| Level | Focus | Repository |
|------|---------|-----------|
| 1 | **Clásico** | `chatbot_classic` |
| 2 | **Embeddings + Sequence** | [chatbot-word2vec](https://github.com/tuusuario/chatbot-word2vec) |
| 3 | **Transformers** | [chatbot-bert](https://github.com/tuusuario/chatbot-bert) |
| 4 | **LLM Integration** | [chatbot-llm](https://github.com/tuusuario/chatbot-llm) |

## 💻 ```Setup & Run```

### Prerequisites

```bash
git clone https://github.com/DanniRodrJ/chatbot_classic.git
cd chatbot_classic
pip install -r requirements.txt
python training_chatbot.py
streamlit run app.py
```

> **NLTK** data is automatically downloaded on first import.

## 🔄 ```Retraining```

```bash
# Edit data/documentacion.json
python training_chatbot.py  # Overwrites models/
...
913/913 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.7361 - loss: 0.5940  
Epoch 194/200
913/913 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.7416 - loss: 0.5875  
Epoch 195/200
913/913 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.7326 - loss: 0.5919  
Epoch 196/200
913/913 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.7400 - loss: 0.5924  
Epoch 197/200
913/913 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.7448 - loss: 0.5758  
Epoch 198/200
913/913 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.7337 - loss: 0.5844  
Epoch 199/200
913/913 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.7324 - loss: 0.5892  
Epoch 200/200
913/913 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.7372 - loss: 0.5880  
Model trained and saved.
```

## 👩‍💻 ```Developer```

If you would like to contact me, simply click on my name and you will be redirected to my LinkedIn profile. I would be delighted 🤗 to answer your questions and share more details about my skills and experience.

<div align="center">

*AI Engineer*
| [<img src="https://avatars.githubusercontent.com/u/123108361?v=4" width=115><br><sub>Danniela Rodríguez</sub>](https://www.linkedin.com/in/danniela-rodriguez-jove-/)
| :---: |

<div align="left">

## 🙌 ```Acknowledgements and Updates```

*Thank you for reviewing this project* 🤗! *If you would like to stay informed about future updates, please star the repository* ⭐. *You can find the option to do so at the top right of the page. Your support is greatly appreciated.*