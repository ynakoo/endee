# Multi-Personality AI Chatbot with Memory

This barebones conversational AI uses the **Endee** vector database for local memory retrieval (RAG) and the **Groq API** to power 4 distinct fallback personalities.

## Key Points & Architecture
1. **System Architecture**:
   - **Frontend/CLI**: `app.py` main chat loop.
   - **Embedder**: `core/embedding_model.py` uses `sentence-transformers` (`all-MiniLM-L6-v2`) to create 384-dimensional dense vectors locally.
   - **Vector Database**: Endee (Local, self-hosted via `localhost:8080`) stores embeddings and metadata. It uses `MessagePack` over HTTP for fast vector search operations.
   - **LLM Engine**: `core/llm_engine.py` requests chat completion from Groq API (e.g. `llama3-8b-8192`), dynamically injecting retrieved context + current persona instructions.
   
2. **Setup Instructions**:
   - Ensure the Endee vector DB server is running on port 8080 (e.g., `cd /path/to/endee && ./run.sh`).
   - Navigate to `demo_app` and run `pip install -r requirements.txt`.
   - Your `.env` file should contain your `GROQ_API_KEY`.
   - Run the chatbot using `python app.py`.

3. **How the RAG Pipeline Works**:
   - The user inputs a message. The `sentence-transformers` model creates a 384d vector representing its semantic meaning.
   - This vector is passed to Endee through `/api/v1/index/chat_memory/search` to retrieve the Top-3 past dialogue snippets.
   - The retrieved context + the current personality (mentor, friend, etc.) + the original input is sent to the LLM via Groq.
   - Following the LLM's response, the new (user_query, bot_response) pair is embedded and saved entirely in Endee (`/api/v1/index/chat_memory/vector/insert`) for future context!
   
4. **Performance & Improvement Suggestions**:
   - **Sparse + Dense (Hybrid Search)**: Endee supports sparse vectors. We could use BM25 to combine exact-keyword and semantic matches for better accuracy.
   - **Chunking Strategy**: For very long replies, we could chunk the metadata for better search granularity.
   - **Prefiltering**: We can assign "user_id" or "chat_session" inside Endee's metadata/filters to strictly divide contexts.
