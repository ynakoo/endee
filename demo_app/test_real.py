import unittest
from unittest.mock import patch
import logging
import time

# Suppress sentence-transformers logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

import app

def test_conversation():
    # We will wait for Endee to be completely ready
    time.sleep(2)
    inputs = [
        "Hi! Describe who you are.",             # Talk to Mentor
        "/switch", "friend",                     # Switch to friend
        "I'm feeling a bit anxious about my new project. It's an AI chatbot.",
        "/switch", "interviewer",
        "Can you ask me a question about my project?",
        "/exit"
    ]
    
    def mock_input(prompt):
        if inputs:
            time.sleep(1.5) # Wait for Endee to index the previous insertion
            val = inputs.pop(0)
            print(f"{prompt}{val}")
            return val
        return "/exit"

    print("Running REAL test against Endee vector database...")
    app.INDEX_NAME = f"test_chat_{int(time.time())}"
    with patch("builtins.input", side_effect=mock_input):
        app.main()

if __name__ == "__main__":
    test_conversation()
