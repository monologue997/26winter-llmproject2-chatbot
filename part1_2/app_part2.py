import streamlit as st
from openai import OpenAI

# Set the title of the web app
st.title("Mini Project 2: Streamlit Chatbot")

# ----------------------------------------------------------------
# 1. Initialize Client & Configuration
# ----------------------------------------------------------------

# Access the API Key securely. 
# For local development, create a .streamlit/secrets.toml file.
api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# ----------------------------------------------------------------
# 2. Define Helper Functions
# ----------------------------------------------------------------

def get_conversation() -> str:
    """
    Iterates over all the messages in the session state and concatenates them 
    into a single string. (Useful for Part-3 context handling)
    """
    conversation_string = ""
    if "messages" in st.session_state:
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            conversation_string += f"{role}: {content}\n"
    return conversation_string

# ----------------------------------------------------------------
# 3. Initialize Session State
# ----------------------------------------------------------------

# Initialize the OpenAI model (using gpt-3.5-turbo as requested)
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize the chat history list if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------------------------------------
# 4. Display Conversation History
# ----------------------------------------------------------------

# Iterate through the message history and display each one
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------------------------------------------
# 5. Handle User Input & Generate Response
# ----------------------------------------------------------------

# st.chat_input creates the text box at the bottom
if prompt := st.chat_input("What would you like to chat about?"):
    
    # 5.1 Display user message in the chat interface
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 5.2 Add user message to session state history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 5.3 Generate Assistant Response
    with st.chat_message("assistant"):
        # Create a placeholder to stream the response (optional but looks nice) or just wait
        message_placeholder = st.empty()
        full_response = ""
        
        # Call OpenAI API with the full conversation history
        # This allows the bot to "remember" what you said previously
        try:
            response = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=False # Set to True if you want typing effect, kept False for simplicity
            )
            
            full_response = response.choices[0].message.content
            st.markdown(full_response)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # 5.4 Add assistant response to session state history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # (Optional) Verify helper function works by printing to terminal
    # print(get_conversation())