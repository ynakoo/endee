import unittest
from unittest.mock import patch, MagicMock
import msgpack
import json
import logging
from config.personalities import PERSONALITIES

# We suppress some logging from sentence-transformers for clean output
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

import app

class MockResponse:
    def __init__(self, status_code, json_data=None, content=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.content = content
        self.text = text
    def json(self):
        return self._json

# Mock memory storage
mock_db = []

def mocked_requests(*args, **kwargs):
    url = args[0]
    if "health" in url:
        return MockResponse(200, text="OK")
    elif "index/create" in url:
        return MockResponse(200, text="Created")
    elif "vector/insert" in url:
        payload = kwargs.get("json", [])
        if payload:
            mock_db.extend(payload)
        return MockResponse(200, text="Inserted")
    elif "search" in url:
        # Simulate returning top 1 from mock_db if it has items
        if mock_db:
            # We just return the last inserted item as messagepack for demo
            # The structure app.py expects: we can just pack a list
            result = [{"meta": mock_db[-1]["meta"]}]
            packed = msgpack.packb(result)
            return MockResponse(200, content=packed)
        else:
            return MockResponse(200, content=msgpack.packb([]))
    
    return MockResponse(404, text="Not Found")


@patch("requests.get", side_effect=mocked_requests)
@patch("requests.post", side_effect=mocked_requests)
def test_conversation(mock_post, mock_get):
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
            val = inputs.pop(0)
            print(f"{prompt}{val}")
            return val
        return "/exit"

    with patch("builtins.input", side_effect=mock_input):
        app.main()

if __name__ == "__main__":
    test_conversation()
