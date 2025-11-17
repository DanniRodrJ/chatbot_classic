from src.chatbot import Chatbot
from src.utils.utils import extract_order_number
import streamlit as st

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

chatbot = Chatbot()

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
        
