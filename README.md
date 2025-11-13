# Chatbot Clásico con Red Neuronal (MLP + BoW)

> Un **chatbot pre-LLM** con:
>
> - Bolsa de palabras + lematización
> - Red neuronal **MLP** (3 capas)
> - 11 intenciones reales
> - Contexto simple (última intención)
> - Interfaz con Streamlit

---

## Demo

![Demo](demo.gif)

---

## Tecnologías

- **Python**
- **TensorFlow / Keras** → Red Neuronal MLP
- **NLTK** → Lematización, WordNet, sinónimos
- **Streamlit** → Interfaz estilo Grok
- **BoW (Bag of Words)** → Representación de texto

---

## Arquitectura del Proyecto

```bash
chatbot_classic/
│
├── src/
│   ├── init.py
│   ├── config.py        
│   ├── model.py          ← Entrenamiento
│   ├── chatbot.py        ← Inferencia
│   └── utils.py          ← Preprocesamiento
│
├── data/
│   └── documentacion.json
│
├── app.py                ← Streamlit
├── training_chatbot.py   ← Entrada para entrenar
├── requirements.txt
└── README.md
```

## Comparativa: Clásico vs LLM

| Característica       | Este Chatbot (MLP)         | LLM (Grok, GPT)               |
|----------------------|----------------------------|-------------------------------|
| **Red Neuronal**     | MLP (128 → 64 → N)         | Transformer (175B parámetros)  |
| **Contexto**         | 1 frase                    | 128k tokens                   |
| **Entrenamiento**    | 200 épocas, <1MB           | Billones de tokens            |
| **Flexibilidad**     | Baja (intenciones fijas)   | Alta (responde cualquier cosa)|
| **Costo**            | $0                         | $$$$$                         |
| **Explicabilidad**   | 100%                       | 0%                            |

---

## Limitaciones del Modelo (Transparencia Técnica)

| Limitación | Descripción | Impacto |
|-----------|-------------|--------|
| **Sin contexto real** | Solo analiza 1 frase a la vez | No recuerda el flujo |
| **Clasificación rígida** | Solo responde a intenciones predefinidas | Falla con variaciones |
| **BoW pierde orden** | `"no quiero"` = `"quiero no"` | Errores semánticos |
| **Sin aprendizaje en tiempo real** | Requiere reentrenar | No se adapta |
| **No entiende entidades complejas** | Números, fechas, nombres | Necesita regex |

> **Este no es un bug, es una característica**:  
> _"Muestra exactamente cómo funcionaban los chatbots antes de los LLMs"_

---

## Evolución del Proyecto (Visita mis repos)

| Nivel | Proyecto | Tecnología | Enlace |
|------|---------|-----------|-------|
| 1 | **Clásico** | MLP + BoW | `este-repo` |
| 2 | **Embeddings** | Word2Vec + LSTM | [chatbot-word2vec](https://github.com/tuusuario/chatbot-word2vec) |
| 3 | **Transformers** | BERT + Fine-tuning | [chatbot-bert](https://github.com/tuusuario/chatbot-bert) |
| 4 | **LLM** | Llama 3 / Grok API | [chatbot-llm](https://github.com/tuusuario/chatbot-llm) |

> **LLMs entran en el nivel 4** → cuando usas modelos preentrenados con **billones de parámetros**.

---

## Instalación

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('wordnet')"
python training_chatbot.py
streamlit run app.py
```
