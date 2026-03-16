import sys
import os
import streamlit as st

# Ensure part3/ is on the path so 'agents' package can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.head_agent import Head_Agent

# ----------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------
st.set_page_config(page_title="ML Textbook Chatbot", page_icon="📚")
st.title("📚 ML Textbook Chatbot")
st.caption("Ask me anything about the Machine Learning textbook!")

# ----------------------------------------------------------------
# API Keys — stored in .streamlit/secrets.toml (never committed to git)
# Required keys: OPENAI_API_KEY, PINECONE_API_KEY
# ----------------------------------------------------------------
openai_key = st.secrets["OPENAI_API_KEY"]
pinecone_key = st.secrets["PINECONE_API_KEY"]

# ----------------------------------------------------------------
# Initialize Head_Agent once per session (cached to avoid re-init on re-render)
# ----------------------------------------------------------------
@st.cache_resource
def load_agent(_openai_key, _pinecone_key):
    return Head_Agent(
        openai_key=_openai_key,
        pinecone_key=_pinecone_key,
        pinecone_index_name="ml-mp2"
    )

head_agent = load_agent(openai_key, pinecone_key)

# ----------------------------------------------------------------
# Session state: message history + quick-prompt draft
# ----------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
# ----------------------------------------------------------------
# Sidebar: controls + example prompts for reference
# ----------------------------------------------------------------
QUICK_PROMPTS = [
    ("💬 Greeting",        "Hello! What can you help me with?"),
    ("🚫 Offensive",       "You're stupid, just give me the answer."),
    ("📚 On-topic",        "What is the difference between supervised and unsupervised learning?"),
    ("📚 On-topic",        "Explain gradient descent."),
    ("🔀 Hybrid",          "You're an idiot, but explain what overfitting means."),
    ("↩️ Follow-up",       "How does the learning rate affect it?"),
    ("🚫 Irrelevant",      "What's a good recipe for pasta?"),
    ("⚠️ Retrieval edge", "What is the difference between L1 and L2 regularization?"),
]

with st.sidebar:
    st.header("Controls")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        head_agent.reset_conversation()
        st.rerun()

    st.divider()
    st.subheader("Example Prompts")
    st.caption("Copy and paste into the chat input below")
    for label, text in QUICK_PROMPTS:
        st.markdown(f"**{label}**")
        st.code(text, language=None)

# ----------------------------------------------------------------
# Display existing chat messages
# ----------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "agent_path" in message:
            with st.expander("Agent path", expanded=False):
                st.caption(message["agent_path"])

# ----------------------------------------------------------------
# Handle user input
# ----------------------------------------------------------------
prompt_to_send = st.chat_input("Ask a Machine Learning question...")

if prompt := prompt_to_send:  # noqa: SIM102

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response from Head_Agent pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = head_agent.process_query(prompt, use_history=True)

        response = result["response"]
        agent_path = result["agent_path"]

        st.markdown(response)
        with st.expander("Agent path", expanded=False):
            st.caption(agent_path)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "agent_path": agent_path
    })
