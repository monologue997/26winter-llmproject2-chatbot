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
if "draft_prompt" not in st.session_state:
    st.session_state.draft_prompt = ""

# ----------------------------------------------------------------
# Sidebar: controls + quick prompts
# ----------------------------------------------------------------
QUICK_PROMPTS = [
    ("💬 Greeting", "Hello! What can you help me with?"),
    ("🚫 Offensive input", "You're stupid, just give me the answer."),
    ("📚 On-topic (RAG)", "What is the difference between supervised and unsupervised learning?"),
    ("⚡ On-topic (RAG)", "Explain gradient descent."),
    ("🔀 Hybrid (mixed intent)", "You're an idiot, but explain what overfitting means."),
    ("↩️ Follow-up", "How does the learning rate affect it?"),
    ("🚫 Irrelevant", "What's a good recipe for pasta?"),
]

with st.sidebar:
    st.header("Controls")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        head_agent.reset_conversation()
        st.rerun()

    st.divider()
    st.subheader("Quick Prompts")
    st.caption("Click to load into editor below")
    for label, text in QUICK_PROMPTS:
        if st.button(label, key=f"qp_{label}", use_container_width=True):
            st.session_state.draft_prompt = text
            st.rerun()

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
# Quick-prompt editor: shows only when a quick prompt is loaded
# ----------------------------------------------------------------
prompt_to_send = None

if st.session_state.draft_prompt:
    with st.container(border=True):
        edited = st.text_area(
            "Edit before sending:",
            value=st.session_state.draft_prompt,
            height=80,
            key="draft_editor",
            label_visibility="collapsed",
        )
        col_send, col_cancel, _ = st.columns([1, 1, 4])
        with col_send:
            if st.button("Send", type="primary"):
                prompt_to_send = edited
                st.session_state.draft_prompt = ""
        with col_cancel:
            if st.button("Cancel"):
                st.session_state.draft_prompt = ""
                st.rerun()

# ----------------------------------------------------------------
# Handle user input (normal chat input or quick-prompt send)
# ----------------------------------------------------------------
if not prompt_to_send:
    prompt_to_send = st.chat_input("Ask a Machine Learning question...")

if prompt := prompt_to_send:

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
