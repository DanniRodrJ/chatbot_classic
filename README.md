# Classic Intent Classifier: BoW + MLP (Non-LLM NLU)

![image](/assets/image.png)

A fully controllable, explainable intent classification system using **Bag of Words (BoW)** and a **Multilayer Perceptron (MLP)**.This system is optimized for customer service centers with a fixed set of intents, providing a **low-cost**, highly transparent solution.

## 🚀 ```Demo```

![Demo](/assets/demo.gif)
→ Try the [Classic Intent Classifier](https://chatbotclassic-dannirodrj.streamlit.app/)

## ⚙️ ```Tech Stack```

| Technology | Purpose |
|----------|--------|
| **Python** | Core language |
| **TensorFlow / Keras** | MLP model training & inference |
| **NLTK** | Tokenization, lemmatization, synonym expansion |
| **Streamlit** | Production-ready local UI |
| **Plotly 3D** | Neural network visualization|
| **BoW** | Text-to-vector encoding |

## 🔬 ```Technical Overview```

| Component | Implementation |
|---------|----------------|
| **Model** | MLP (128 → 64 → N) with Dropout |
| **Feature Engineering** | BoW + lemmatization |
| **Preprocessing** | NLTK (tokenization, lemmatization) |
| **Context** | Previous intent state (`last_intent`) |
| **Entity Extraction** | **Regex + Slot Filling** |
| **Framework** | TensorFlow/Keras |
| **Interface** | Streamlit (local deployment) |
| **Architecture** | Modular (`src/`, `data/`, `models/`) |

## 📁 ```Project Structure```

```bash
chatbot_classic/
├── assets/
│
├── data/
│   ├── intents.json
│   └── intents_test_only.json  # Zero-shot evaluation
│
├── models/                     # Generated artifacts
│   ├── chatbot_model.keras
│   ├── words.pkl
│   └── classes.pkl
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Path management
│   ├── model.py               # Training pipeline
│   ├── chatbot.py             # Inference engine
│   └── utils/
│       └── utils.py           # Preprocessing + NLTK auto-download
│
├── .gitignore
├── app.py                     # Streamlit frontend
├── README.md   
├── evaluate_zero_shot.py
├── requirements.txt
└── training_chatbot.py        # Training entrypoint
```

## ⬇️ ```Inference Pipeline```

The inference engine processes the input to determine intent and fill in key slots, such as order numbers.

```bash
Input: "I want to return a damaged item"
        ↓
Tokenization → Lemmatization → ["want", "return", "damaged", "item"]
        ↓
Bag of Words Vector → [0, 1, 0, 1, ...] (165-dim)
        ↓
MLP (128→64→13) → [0.02, 0.91, 0.01, ...]
        ↓
Predicted intent: "return" (91% confidence)
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
- **Training**: 300 epochs, batch size 5

## 👀 ```Neural Network Visualizer (3D Interactive)```

See **exactly** how your message flows through the neural network:

- Which words activate the BoW vector
- Neuron activations layer by layer
- Final softmax probabilities
- Hover to see words and intents

→ Try the [Neural Network Visualizer](https://chatbotclassic-dannirodrj.streamlit.app/) (second page)

![Neural Visualizer](/assets/neural-visualizer-screenshot.png)

## ⭐ ```Key Features```

|Feature|Description|
|-------|-----------|
|**Intent Classification**|13 predefined intents|
|**Simple Context** | Uses last_intent for conversational flow|
|**Slot Filling (Enhanced)**|Order numbers extracted via regex and used to format the response|
|**Response Deduplication**|Filters repeated responses|
|**Synonym Augmentation**|Automatic pattern expansion via WordNet|
|**Robust Fallback**|Regex + previous intent if confidence < 0.5|

## 📈 ```Training Metrics```

| Metric                          | Value       | Notes                                      |
|---------------------------------|-------------|----------------------------------------------------|
| Raw Intent Accuracy             | **60.0%**   | Best epoch 16 • Early stopping • No synonym noise |
| **Zero-Shot Real-World Accuracy**     | **54.84%**    | 93 completely unseen natural English phrases |
| **Effective Conversational Accuracy** | **>95%** | Context tracking + slot filling + regex fallbacks |
| Inference time                        | **< 5 ms**  | CPU only |
| Model size                            | **~95 KB**  | Tiny & deployable anywhere |
| Cost                                  | **$0**      | 100% local • No APIs |

> **Key insight**:  
> WordNet synonym augmentation **was disabled after testing** — it introduced noise and hurt generalization.  
> Result: **+10% validation** and **+8–10% real-world accuracy**.  
> This is real engineering: **less noise > more data**.

**Zero-shot evaluation**: `python evaluate_zero_shot.py` → 54.84% on 100% unseen phrases.

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
python evaluate_zero_shot.py
```

> **NLTK** data is automatically downloaded on first import.

## 🔄 ```Retraining```

```bash
# Edit data/documentacion.json
python training_chatbot.py  # Overwrites models/
...
766/766 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.7361 - loss: 0.5940  
Epoch 25/300
766/766 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.7416 - loss: 0.5875  
Epoch 26/300
766/766 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.7326 - loss: 0.5919  
Epoch 27/300
766/766 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.7400 - loss: 0.5924  
Epoch 28/300
766/766 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.7448 - loss: 0.5758  
Epoch 29/300
766/766 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.7337 - loss: 0.5844  
Epoch 30/300
766/766 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.7324 - loss: 0.5892  
Epoch 31/300
766/766 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.7372 - loss: 0.5880  
Epoch 31: early stopping
Restoring model weights from the end of the best epoch: 16.
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
