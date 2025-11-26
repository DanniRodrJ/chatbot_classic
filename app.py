from src.chatbot import Chatbot
from src.utils.utils import extract_order_number, clean_sentence, get_synonyms
import streamlit as st
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import numpy as np

st.set_page_config("Classic Intent Classifier", layout='wide', page_icon="robot", initial_sidebar_state="expanded")
st.theme = "auto"

st.markdown("""
<style>
    .stApp {
        background: var(--background-color) !important;
        color: var(--text-color) !important;
    }

    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stAppViewContainer"] > .main {
        background: var(--background-color) !important;
    }

    section[data-testid="stChatInput"],
    section[data-testid="stChatInput"] input,
    .stChatInput > div > div > input {
        background: var(--background-color) !important;
        color: var(--text-color) !important;
        border-color: var(--text-color) !important;
    }

    [data-testid="stChatMessageUser"] {
        background: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
    }

    [data-testid="stChatMessageAssistant"] {
        background: #2f855a !important;
        color: white !important;
    }

    .main-title {
        text-align: center;
        color: var(--text-color);
        font-size: 28px;
        margin: 20px 0 8px;
        font-weight: 600;
    }
    .subtitle {
        text-align: center;
        color: var(--text-color);
        opacity: 0.8;
        font-size: 14px;
        margin-bottom: 30px;
    }

    .stHeader, .stFooter, header { 
        display: none !important; 
    }

    .block-container {
        max-width: 700px !important;
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.image("https://img.icons8.com/fluency/100/brain.png", width=100)
st.sidebar.markdown("### Navegation")
page = st.sidebar.radio("Go to:", ["Chatbot", "Neural Network Visualizer"], index=0)

chatbot = Chatbot()

if page == "Chatbot":
    st.markdown("<h1 class='main-title'>Classic Intent Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>BoW + MLP • Non-LLM NLU • Full Control</p>", unsafe_allow_html=True)

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
        
        res = chatbot.get_response(insts,prompt, last_response,order_number=order_num_slot)
        
        with st.chat_message("assistant"):
            st.markdown(res)
        st.session_state.messages.append({"role": "assistant", "content": res})
        
        if insts and insts[0]['intent'] != 'not_understood':
            st.session_state.last_intent = insts[0]['intent']

else:
    st.markdown("""
    <style>
        .block-container {max-width: 60vw !important; padding: 2rem !important;}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='main-title'>Neural Network Visualizer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>See how your message flows through the MLP • 3D Interactive</p>", unsafe_allow_html=True)

    sentence = st.text_area("Enter a message:", value="I want to return a product", height=80, label_visibility="collapsed")

    if st.button("Visualize Forward Pass", type="primary", use_container_width=True):
        with st.spinner("Processing entire network..."):
            sentence_clean = clean_sentence(sentence)

            bow = [0] * len(chatbot.words)
            activated_indices = []
            for w in sentence_clean:
                for i, word in enumerate(chatbot.words):
                    if word == w or word in get_synonyms(w):
                        if bow[i] == 0:
                            bow[i] = 1
                            activated_indices.append(i)
                        break

            bow = np.array(bow, dtype=np.float32)
            input_vec = np.expand_dims(bow, axis=0)

            x = input_vec
            layer_activations = [bow.tolist()]
            for layer in [l for l in chatbot.model.layers if isinstance(l, tf.keras.layers.Dense)]:
                w, b = layer.get_weights()
                z = tf.matmul(x, w) + b
                if layer.activation == tf.keras.activations.relu:
                    x = tf.nn.relu(z)
                elif layer.activation == tf.keras.activations.softmax:
                    x = tf.nn.softmax(z, axis=-1)
                else:
                    x = z
                layer_activations.append(x.numpy()[0].tolist())

            input_full = layer_activations[0]
            hidden_128 = layer_activations[1]
            hidden_64  = layer_activations[2]
            output_13  = layer_activations[3]

            pred_idx = np.argmax(output_13)
            predicted = chatbot.classes[pred_idx]

            st.success(f"**Prediction:** `{predicted}` | **Confidence:** {output_13[pred_idx]:.1%}")
            active_words = [chatbot.words[i] for i in activated_indices]
            st.warning(f"**Activated words ({len(active_words)}):** {', '.join(active_words)}")
            st.info(f"**Total vocabulary:** {len(chatbot.words)} words")
            st.divider()

            word_in_input = [""] * len(input_full)
            for idx in activated_indices:
                word_in_input[idx] = chatbot.words[idx]

            #input_rows, input_cols = 42, 40

            layers = [
                {"acts": input_full, "grid": (14, 15), "z": 0, "name": "Input (206)"},
                {"acts": hidden_128, "grid": (16, 8), "z": 200, "name": "Hidden 128"},
                {"acts": hidden_64,  "grid": (8, 8), "z": 400, "name": "Hidden 64"},
                {"acts": output_13,  "grid": (1, 18), "z": 600, "name": "Output 18"}
            ]


            fig = go.Figure()

            NEURON_SIZE = 9

            for l, layer in enumerate(layers):
                acts = layer["acts"]
                rows, cols = layer["grid"]
                z_pos = layer["z"]

                if l == 0:
                    spacing = 5.0
                elif l == 1:
                    spacing = 8.0
                elif l == 2:
                    spacing = 12.0
                else:
                    spacing = 28.0

                for r in range(rows):
                    for c in range(cols):
                        idx = r * cols + c
                        if idx >= len(acts): continue
                        val = acts[idx]

                        x = c * spacing - (cols-1)*spacing/2
                        y = r * spacing - (rows-1)*spacing/2

                        is_active = (l == 0 and val == 1.0) or (l > 0 and val > 0.08)
                        color = '#ff3333' if is_active else "#767676"  # #333333
                        opacity = 0.8 if is_active else 0.4

                        hover = f"<b>{layer['name']}</b><br>Neuron {idx}<br>Value: {val:.4f}<br>"
                        if l == 0 and word_in_input[idx]:
                            hover += f"Word: <b>{word_in_input[idx]}</b>"
                        if l == 3:
                            intent = chatbot.classes[idx]
                            hover += f"<br>Intent: <b>{intent}</b>"
                            if idx == pred_idx: hover += " ← FORECAST"

                        fig.add_trace(go.Scatter3d(
                            x=[x], y=[y], z=[z_pos],
                            mode='markers',
                            marker=dict(size=NEURON_SIZE, color=color, opacity=opacity,
                                        line=dict(color='white', width=0.5)),
                            text=hover, hoverinfo='text', showlegend=False
                        ))

            for l in range(3):
                z1, z2 = layers[l]["z"], layers[l+1]["z"]
                acts1, acts2 = layers[l]["acts"], layers[l+1]["acts"]
                r1, c1 = layers[l]["grid"]
                r2, c2 = layers[l+1]["grid"]

                sp1 = 5.0 if l == 0 else 8.0 if l == 1 else 12.0
                sp2 = 8.0 if l == 0 else 12.0 if l == 1 else 28.0

                #active1 = [i for i, v in enumerate(acts1) if v > 0.08][:100]
                active1 = [i for i, v in enumerate(acts1) if (l == 0 and v == 1.0) or (l > 0 and v > 0.08)]
                active1 = active1[:120] if l == 0 else active1

                for i in active1:
                    r1_i, c1_i = divmod(i, c1)
                    x1 = c1_i * sp1 - (c1-1)*sp1/2
                    y1 = r1_i * sp1 - (r1-1)*sp1/2

                    for j in range(len(acts2)):
                        if acts2[j] < 0.08: continue
                        r2_j, c2_j = divmod(j, c2)
                        x2 = c2_j * sp2 - (c2-1)*sp2/2
                        y2 = r2_j * sp2 - (r2-1)*sp2/2

                        fig.add_trace(go.Scatter3d(
                            x=[x1,x2], y=[y1,y2], z=[z1,z2],
                            mode='lines',
                            line=dict(color='#00ffff', width=1),
                            opacity=0.15,
                            hoverinfo='none',
                            showlegend=False
                        ))

            fig.update_layout(
                height=1000,
                scene=dict(
                    bgcolor="black",
                    camera=dict(eye=dict(x=2.2, y=2.2, z=1.6)),
                    xaxis=dict(showbackground=True, backgroundcolor="rgba(5,5,20,0.9)", 
                            gridcolor="rgba(100,150,255,0.15)", showgrid=True, visible=False),
                    yaxis=dict(showbackground=True, backgroundcolor="rgba(5,5,20,0.9)", 
                            gridcolor="rgba(100,150,255,0.15)", showgrid=True, visible=False),
                    zaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(100,200,255,0.3)",
                        zerolinecolor="#eeeeee",
                        zerolinewidth=3,
                        tickvals=[0, 220, 440, 660],
                        ticktext=["Input (206)", "Hidden 128", "Hidden 64", "Output 18"],
                        tickfont=dict(color="#eeeeee", size=12, family="Consolas"),
                        title="Layers"  
                    ),
                    aspectmode='manual',
                    aspectratio=dict(x=2, y=2, z=1.3)
                ),
                paper_bgcolor="black",
                margin=dict(l=0, r=0, t=50, b=0),
                title=dict(
                    text="Neural Network MLP • BoW + 206 words → 128 → 64 → 18 intents",
                    font=dict(color="#00ffff", size=18, family="Courier New"),
                    x=0.5, y=0.95, xanchor="center", yanchor="top"
                )
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption("Rotate: click + drag | Zoom: wheel | Hover: word / intent")