import time
import requests
import json
import msgpack
from colorama import init, Fore, Style
from core.embedding_model import get_embedding
from core.llm_engine import generate_response
from config.personalities import PERSONALITIES

# Initialize colorama
init(autoreset=True)

ENDEE_URL = "http://localhost:8080"
INDEX_NAME = "chat_memory"
VECTOR_DIM = 384  # from all-MiniLM-L6-v2

def extract_meta_from_search(data):
    """
    Extracts the string 'meta' field from the MessagePack parsed data structure.
    Endee returns a ResultSet which is serialized as a list of arrays:
    [ [distance, id, meta_bytes, filter_string, norm, sparse_vector], ... ]
    """
    res = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, list) and len(item) > 2:
                meta = item[2]
                if isinstance(meta, bytes):
                    try:
                        res.append(meta.decode('utf-8', errors='ignore'))
                    except Exception:
                        pass
                elif isinstance(meta, str) and meta:
                    res.append(meta)
    return res

def setup_endee():
    """Ensure Endee server is running and the index exists."""
    try:
        r = requests.get(f"{ENDEE_URL}/api/v1/health", timeout=3)
        if r.status_code != 200:
            print(f"{Fore.RED}Failed to reach Endee. Please start your local Endee instance. (Status: {r.status_code})")
            return False
    except requests.exceptions.RequestException:
        print(f"{Fore.RED}Could not connect to Endee database at {ENDEE_URL}.")
        print(f"{Fore.YELLOW}Make sure you've started the Endee server on port 8080.")
        return False
    
    # Try to list indexes and see if ours exists 
    # Or simply try to create it directly (it's safe if it already exists, it will return an error or ignore)
    create_payload = {
        "index_name": INDEX_NAME,
        "dim": VECTOR_DIM,
        "space_type": "cosine"
    }
    r = requests.post(f"{ENDEE_URL}/api/v1/index/create", json=create_payload)
    if r.status_code == 200 or r.status_code == 201:
        print(f"{Fore.GREEN}Index '{INDEX_NAME}' is ready.")
    elif r.status_code == 400 and "already exists" in r.text.lower():
        print(f"{Fore.GREEN}Index '{INDEX_NAME}' already exists, skipping creation.")
    # We ignore other errors, assuming index might be created already
    return True

def store_memory(user_msg: str, bot_reply: str, embedding: list[float]):
    """Insert context into the vector dataset."""
    doc_id = f"msg_{int(time.time() * 1000)}"
    meta_text = f"User: {user_msg}\nBot: {bot_reply}"
    
    payload = [{
        "id": doc_id,
        "vector": embedding,
        "meta": meta_text
    }]
    
    r = requests.post(
        f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/vector/insert",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    if r.status_code not in (200, 201):
        print(f"{Fore.RED}Failed to store memory snippet. Status {r.status_code}: {r.text}")

def search_memory(embedding: list[float], top_k: int = 3) -> list[str]:
    """Retrieve top-K similar past interactions based on the user's message."""
    payload = {
        "vector": embedding,
        "k": top_k
    }
    try:
        r = requests.post(
            f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/search",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        if r.status_code == 200:
            # Endee returns msgpack binary data for search.
            data = msgpack.unpackb(r.content, raw=False)
            results = extract_meta_from_search(data)
            return results[:top_k]
    except Exception as e:
        print(f"{Fore.RED}Failed to search memory in Endee: {e}")
    return []

def main():
    print(f"{Fore.CYAN}{Style.BRIGHT}=========================================")
    print(f"{Fore.CYAN}{Style.BRIGHT} Multi-Personality Chatbot with RAG Memory ")
    print(f"{Fore.CYAN}{Style.BRIGHT}=========================================")
    
    if not setup_endee():
        return
    
    current_personality = "mentor"
    
    while True:
        print(f"\n{Fore.GREEN}Current Personality: {Fore.YELLOW}{current_personality.upper()}")
        print(f"{Fore.CYAN}[Type '/switch' to change personality, '/clear' to restart memory scope, '/exit' to quit]")
        
        user_input = input(f"{Fore.WHITE}You: ")
        
        if not user_input.strip():
            continue
        
        if user_input.strip() == "/exit":
            print(f"{Fore.MAGENTA}Goodbye!")
            break
        elif user_input.strip() == "/switch":
            print(f"{Fore.YELLOW}Available personalities: {', '.join(PERSONALITIES.keys())}")
            new_p = input(f"Choose personality: ").strip().lower()
            if new_p in PERSONALITIES:
                current_personality = new_p
                print(f"{Fore.GREEN}Switched to {new_p.upper()}.")
            else:
                print(f"{Fore.RED}Invalid personality.")
            continue
        elif user_input.strip() == "/clear":
            print(f"{Fore.YELLOW}Note: Memory in Endee is persistent. This chat scope continues, but this CLI allows a fresh start if modified.")
            continue
        
        # 1. Embed query
        print(f"{Style.DIM}Thinking...{Style.RESET_ALL}", end="\r")
        query_embedding = get_embedding(user_input)
        
        # 2. Search Endee memory
        past_context = search_memory(query_embedding, top_k=3)
        if past_context:
            print(f"{Style.DIM}(Retrieved {len(past_context)} memory items from Endee){Style.RESET_ALL}")
        
        # 3. Generate response using Groq
        prompt = PERSONALITIES[current_personality]
        bot_response = generate_response(prompt, past_context, user_input)
        
        # 4. Display response
        print(f"\n{Fore.MAGENTA}{current_personality.capitalize()}: {Fore.CYAN}{bot_response}")
        
        # 5. Store memory for future
        store_memory(user_input, bot_response, query_embedding)

if __name__ == "__main__":
    main()
