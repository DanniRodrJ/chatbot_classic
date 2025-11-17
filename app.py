from src.chatbot import Chatbot
from src.utils.utils import extract_order_number, clean_sentence, get_synonyms
import streamlit as st
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import json
import urllib.parse

st.set_page_config("Classic Intent Classifier", layout='wide', page_icon="robot")

st.markdown("""
<style>

    .stApp {
        background: #0f0f0f;
        color: #e0e0e0;
    }

    .main-title {
        text-align: center;
        color: #f7fafc;
        font-size: 28px;
        margin: 20px 0 8px;
        font-weight: 600;
    }
    .subtitle {
        text-align: center;
        color: #a0aec0;
        font-size: 14px;
        margin-bottom: 30px;
    }

    [data-testid="stChatMessageUser"] {
        background: #2d3748 !important;
        color: #e2e8f0 !important;
        border-radius: 16px 16px 4px 16px !important;
        padding: 12px 16px !important;
        max-width: 70% !important;
        margin-left: auto !important;
        margin-right: 10% !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        font-size: 15px;
    }

    [data-testid="stChatMessageAssistant"] {
        background: #2f855a !important;
        color: white !important;
        border-radius: 16px 16px 16px 4px !important;
        padding: 12px 16px !important;
        max-width: 70% !important;
        margin-right: auto !important;
        margin-left: 10% !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        font-size: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    [data-testid="stChatMessageAssistant"]::before {
        content: "robot";
        font-size: 18px;
    }

    .stChatInput {
        max-width: 700px !important;
        margin: 0 auto !important;
        padding: 0 20px !important;
    }
    .stChatInput > div > div > input {
        background: #2d3748 !important;
        color: white !important;
        border-radius: 12px !important;
        border: 1px solid #4a5568 !important;
        padding: 12px !important;
        font-size: 15px;
    }

    .stHeader, .stFooter { display: none !important; }

    .block-container {
        max-width: 700px !important;
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>Classic Intent Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>BoW + MLP • Non-LLM NLU • Full Control</p>", unsafe_allow_html=True)

st.sidebar.image("https://img.icons8.com/fluency/100/brain.png", width=100)
st.sidebar.markdown("### Navegación")
page = st.sidebar.radio("Ir a:", ["Chatbot", "Brain Visualizer"], index=0)

chatbot = Chatbot()

if page == "Chatbot":

    if "messages" not in st.session_state: st.session_state.messages = []
    if "first_message" not in st.session_state: st.session_state.first_message = True
    if "last_intent" not in st.session_state: st.session_state.last_intent = None
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if st.session_state.first_message:
        with st.chat_message("assistant"):
            st.markdown("Hello! How can I assist you today?")
            
        st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I assist you today?"})
        st.session_state.first_message = False
        
    if prompt := st.chat_input("Type your message..."):
        
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        last_response = st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 and st.session_state.messages[-2]["role"] == "assistant" else None
        
        insts = chatbot.predict_intent(prompt, st.session_state.last_intent)
        order_num_slot = None
        
        if insts[0]['intent'] == 'order_number' or (last_response and 'order number' in last_response.lower()):
            order_num_slot = extract_order_number(prompt) or prompt.strip()
        
        res = chatbot.get_response(insts, last_response,order_number=order_num_slot)
        
        with st.chat_message("assistant"):
            st.markdown(res)
        st.session_state.messages.append({"role": "assistant", "content": res})
        
        if insts and insts[0]['intent'] != 'not_understood':
            st.session_state.last_intent = insts[0]['intent']

else:
    st.markdown("## Neural Network Visualizer")
    st.markdown("### 3D Interactive View (Plotly - rotate, zoom, hover for details)")

    sentence = st.text_area("Enter a message:", value="I want to return a damaged product", height=80)

    if st.button("Visualize Forward Pass", type="primary", use_container_width=True):
        with st.spinner("Processing BoW → Activations..."):
            # Tu lógica exacta (sin errores de tolist)
            sentence_clean = clean_sentence(sentence)
            bow = [0] * len(chatbot.words)
            for w in sentence_clean:
                for i, word in enumerate(chatbot.words):
                    if word == w or word in get_synonyms(w):
                        bow[i] = 1
                        break
            bow = np.array(bow, dtype=np.float32)
            input_vec = np.expand_dims(bow, axis=0)

            x = input_vec
            activations = [bow.tolist()]

            for layer in [l for l in chatbot.model.layers if isinstance(l, tf.keras.layers.Dense)]:
                w, b = layer.get_weights()
                z = tf.matmul(x, w) + b
                if layer.activation == tf.keras.activations.relu:
                    x = tf.nn.relu(z)
                elif layer.activation == tf.keras.activations.softmax:
                    x = tf.nn.softmax(z, axis=-1)
                else:
                    x = z
                # Fix: siempre a lista
                act = x.numpy()[0].tolist() if hasattr(x, 'numpy') else x[0].tolist()
                activations.append(act)

            probs = chatbot.model.predict(input_vec, verbose=0)[0]
            pred_idx = np.argmax(probs)
            predicted = chatbot.classes[pred_idx]
            confidence = float(probs[pred_idx])

            active_words = [chatbot.words[i] for i, val in enumerate(bow) if val == 1][:8]  # Limit 8 para viz

        col1, col2 = st.columns([1, 2])
        with col1:
            st.success(f"**Predicted Intent:** `{predicted}`")
            st.info(f"**Confidence:** {confidence:.1%}")
            st.warning(f"**Active Words:** {', '.join(active_words) if active_words else 'None'}")
            st.markdown("### Top 3 Intents")
            for intent, prob in sorted(zip(chatbot.classes, probs), key=lambda x: -x[1])[:3]:
                st.write(f"- `{intent}` → {prob:.1%}")

        with col2:
            import plotly.graph_objects as go
            import numpy as np

            # --- Forward pass ---
            active_indices = np.where(bow == 1)[0]
            active_words = [chatbot.words[i] for i in active_indices[:16]]  # grid 4x4 para input

            x = input_vec
            activations = [bow.tolist()[:16]]  # Input grid 4x4

            for layer in [l for l in chatbot.model.layers if isinstance(l, tf.keras.layers.Dense)]:
                w, b = layer.get_weights()
                z = tf.matmul(x, w) + b
                if layer.activation == tf.keras.activations.relu:
                    x = tf.nn.relu(z)
                elif layer.activation == tf.keras.activations.softmax:
                    x = tf.nn.softmax(z, axis=-1)
                else:
                    x = z
                act = x.numpy()[0].tolist()
                # Hidden: 64 neuronas visibles (grid 8x8)
                visible = act[:64] if len(act) >= 64 else act + [0]*(64-len(act))
                activations.append(visible)

            # Output: 10 intents (fila)
            output_probs = activations[-1][:10] if len(activations[-1]) >= 10 else activations[-1] + [0]*10

            # --- Capas: input (4x4 grid), hidden (8x8 grid), output (1x10 fila) ---
            layers = [activations[0], activations[1], output_probs]
            layer_names = ["Input (palabras)", "Hidden (128→64 visibles)", "Output (intents)"]
            layer_z = [0, 60, 120]  # separación 3D

            # Grids por capa
            grid_sizes = [[4, 4], [8, 8], [1, 10]]  # filas x columnas

            fig = go.Figure()

            x_coords, y_coords, z_coords = [], [], []
            sizes, colors, hover_texts = [], [], []

            for l, (acts, grid, z_pos, name) in enumerate(zip(layers, grid_sizes, layer_z, layer_names)):
                rows, cols = grid
                act_flat = acts[:rows*cols]
                spacing_x = 4 if l == 0 else 2 if l == 1 else 1
                spacing_y = 4 if l == 0 else 2 if l == 1 else 1

                for row in range(rows):
                    for col in range(cols):
                        idx = row * cols + col
                        if idx >= len(act_flat):
                            val = 0
                        else:
                            val = act_flat[idx]

                        x_pos = col * spacing_x - (cols - 1) * spacing_x / 2
                        y_pos = row * spacing_y - (rows - 1) * spacing_y / 2
                        z_pos_layer = z_pos

                        # Color: gris o rojo
                        color = 'red' if val > 0.1 else 'gray'
                        size = 12  # mismo tamaño

                        # Hover
                        hover = f"<b>{name}</b><br>Neurona ({row},{col}) | Act: {val:.3f}<br>"
                        if l == 0 and idx < len(active_words):
                            hover += f"Palabra: <b>{active_words[idx]}</b>"
                        if l == 2:
                            intent = chatbot.classes[idx] if idx < len(chatbot.classes) else f"intent_{idx}"
                            hover += f"Intent: <b>{intent}</b> ({val:.1%})"

                        x_coords.append(x_pos)
                        y_coords.append(y_pos)
                        z_coords.append(z_pos_layer)
                        sizes.append(size)
                        colors.append(color)
                        hover_texts.append(hover)

            # Neuronas
            fig.add_trace(go.Scatter3d(
                x=x_coords, y=y_coords, z=z_coords,
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors,
                    colorscale=[[0, 'gray'], [1, 'red']],  # solo 2 colores
                    opacity=0.9,
                    line=dict(color='white', width=1)
                ),
                text=hover_texts,
                hoverinfo='text',
                name="Neuronas"
            ))

            # --- Conexiones entre capas (solo activas, en grid) ---
            for l in range(len(layers)-1):
                z1, z2 = layer_z[l], layer_z[l+1]
                grid1, grid2 = grid_sizes[l], grid_sizes[l+1]
                rows1, cols1 = grid1
                rows2, cols2 = grid2
                acts1, acts2 = layers[l], layers[l+1]

                spacing_x1 = 4 if l == 0 else 2
                spacing_y1 = 4 if l == 0 else 2
                spacing_x2 = 2 if l == 0 else 1
                spacing_y2 = 2 if l == 0 else 1

                for r1 in range(rows1):
                    for c1 in range(cols1):
                        idx1 = r1 * cols1 + c1
                        if idx1 >= len(acts1) or acts1[idx1] < 0.1: continue

                        x1 = c1 * spacing_x1 - (cols1 - 1) * spacing_x1 / 2
                        y1 = r1 * spacing_y1 - (rows1 - 1) * spacing_y1 / 2

                        for r2 in range(rows2):
                            for c2 in range(cols2):
                                idx2 = r2 * cols2 + c2
                                if idx2 >= len(acts2) or acts2[idx2] < 0.1: continue

                                x2 = c2 * spacing_x2 - (cols2 - 1) * spacing_x2 / 2
                                y2 = r2 * spacing_y2 - (rows2 - 1) * spacing_y2 / 2

                                strength = min(acts1[idx1], acts2[idx2])

                                fig.add_trace(go.Scatter3d(
                                    x=[x1, x2], y=[y1, y2], z=[layer_z[l], layer_z[l+1]],
                                    mode='lines',
                                    line=dict(color='cyan', width=1.5),
                                    opacity=strength * 0.6,
                                    hoverinfo='none',
                                    showlegend=False
                                ))

            # --- Estilo 3D inmersivo ---
            fig.update_layout(
                title="Red Neuronal MLP - 3D Interactiva (grid real)",
                scene=dict(
                    xaxis=dict(title="X", showgrid=False, zeroline=False),
                    yaxis=dict(title="Y", showgrid=False, zeroline=False),
                    zaxis=dict(title="Capas", tickvals=layer_z, ticktext=layer_names),
                    bgcolor="black",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))  # vista diagonal 3D real
                ),
                paper_bgcolor="black",
                font=dict(color="white"),
                height=750,
                margin=dict(l=0, r=0, t=60, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption("3D real: rotar para ver profundidad | Hover para palabra/intent | Conexiones solo activas")