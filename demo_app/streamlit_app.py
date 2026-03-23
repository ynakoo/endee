import streamlit as st
import time
from core.embedding_model import get_embedding
from core.llm_engine import generate_response
from config.personalities import PERSONALITIES

# We must import app.py to reuse its Endee functions
import app

# Customize page layout
st.set_page_config(page_title="Endee Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 Multi-Personality AI Chatbot (with Endee Memory)")

# Initialize session state for messages and personality
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    
    # Check Endee health
    endee_ready = app.setup_endee()
    if endee_ready:
        st.success("✅ Endee Local Vector DB Connected")
    else:
        st.error("❌ Endee Local Vector DB Offline (Please start `./run.sh`)")

    # Personality selector
    st.subheader("Select Personality")
    selected_persona = st.selectbox(
        "Who would you like to talk to?",
        options=list(PERSONALITIES.keys()),
        index=list(PERSONALITIES.keys()).index("mentor")
    )
    
    if st.button("Clear Conversation History"):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("Your messages are vectorized locally utilizing `sentence-transformers` and stored in the **Endee** local vector database. When you send a message, it searches past vector entries, and sends the top matches + the personality config to Groq's LLM API!")

# Display chat messages from session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Accept user input
if user_input := st.chat_input("Type your message here..."):
    
    # 1. Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Proceed if Endee is ready
    if not endee_ready:
        st.error("Cannot proceed. Endee isn't running on localhost:8080.")
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking as {selected_persona.capitalize()}..."):
            
            # 2. Embed input
            query_embedding = app.get_embedding(user_input)

            # 3. Search Endee memory
            past_context = app.search_memory(query_embedding, top_k=3)
            
            # Show retrieved context in expander
            if past_context:
                with st.expander(f"🧠 Retrieved {len(past_context)} memories from Endee DB"):
                    for c in past_context:
                        st.text(c)

            # 4. Generate response using Groq
            prompt = PERSONALITIES[selected_persona]
            response = generate_response(prompt, past_context, user_input)
            
            # Display response
            st.markdown(response)

            # 5. Store new context to Endee
            app.store_memory(user_input, response, query_embedding)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
