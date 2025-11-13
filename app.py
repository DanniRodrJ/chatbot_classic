import streamlit as st
from chatbot import predict_class, get_response, intents

# Configuración de Página
st.set_page_config("Asistente Virtual", layout='wide', page_icon="robot")

st.markdown("""
<style>
    /* Fondo oscuro */
    .stApp {
        background: #0f0f0f;
        color: #e0e0e0;
    }

    /* Título centrado */
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

    /* Mensajes del usuario */
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

    /* Mensajes del asistente */
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

    /* Input centrado y compacto */
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

    /* Ocultar header y footer */
    .stHeader, .stFooter { display: none !important; }

    /* Centrar todo el contenido */
    .block-container {
        max-width: 700px !important;
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>Asistente Virtual</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Chatbot clásico • Red Neuronal • Pre-LLM</p>", unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []
if "first_message" not in st.session_state: st.session_state.first_message = True
if "last_intent" not in st.session_state: st.session_state.last_intent = None
    
# Mostrar el historico de los mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# Mensaje inicial
if st.session_state.first_message:
    with st.chat_message("assistant"):
        st.markdown("Hola, ¿Cómo puedo ayudarte?")
        
    st.session_state.messages.append({"role": "assistant", "content": "Hola, ¿Cómo puedo ayudarte?"})
    st.session_state.first_message = False
    
# Creacion del promt
if prompt := st.chat_input("Escribe tu mensaje..."):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Última respuesta (para evitar repetición)
    last_response = st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 and st.session_state.messages[-2]["role"] == "assistant" else None
    
    # Implementacion del algoritmo de IA
    insts = predict_class(prompt, st.session_state.last_intent)
    res = get_response(insts, intents, st.session_state.last_intent, last_response)
    
    
    if insts[0]['intent'] == 'numero_pedido' and st.session_state.last_intent == 'devolucion':
        # Extraer número con regex
        match = re.search(r'\b\d{5,}\b', prompt)
        order_num = match.group(0) if match else prompt.strip()
        res = res.format(order_num)
    
    
    with st.chat_message("assistant"):
        st.markdown(res)
    st.session_state.messages.append({"role": "assistant", "content": res})
    
    # Guardar intención
    if insts and insts[0]['intent'] != 'no_entendido':
        st.session_state.last_intent = insts[0]['intent']
        
