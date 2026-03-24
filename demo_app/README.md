
# Multi-Personality AI Chatbot with Memory

This barebones conversational AI uses the **Endee** vector database for local memory retrieval (RAG) and the **Groq API** to power 4 distinct fallback personalities.

## Key Points & Architecture
1. **System Architecture**:
   - **Frontend/CLI**: `app.py` main chat loop.
   - **Embedder**: `core/embedding_model.py` uses `sentence-transformers` (`all-MiniLM-L6-v2`) to create 384-dimensional dense vectors locally.
   - **Vector Database**: Endee (Local, self-hosted via `localhost:8080`) stores embeddings and metadata. It uses `MessagePack` over HTTP for fast vector search operations.
   - **LLM Engine**: `core/llm_engine.py` requests chat completion from Groq API (e.g. `llama3-8b-8192`), dynamically injecting retrieved context + current persona instructions.
   
## Setup Instructions

### 1. Start the Endee Vector Database
Before running the chatbot, the Endee server must be running on port **8080**.

- **Apple Silicon (M1/M2/M3/M4):**
  ```bash
  cd ..
  chmod +x ./install.sh ./run.sh
  ./install.sh --release --neon
  ./run.sh
  ```
- **Intel/AMD Mac or Linux:**
  ```bash
  cd ..
  ./install.sh --release --avx2
  ./run.sh
  ```

### 2. Install Python Dependencies
Open a **new terminal window** and install the required libraries:

```bash
cd demo_app
pip3 install -r requirements.txt
pip3 install streamlit
```

### 3. Configure API Keys
The chatbot requires a **Groq API Key** for LLM responses.

- Create a `.env` file in the `demo_app` directory:
  ```bash
  touch .env
  ```
- Add your key:
  ```text
  GROQ_API_KEY=your_groq_api_key_here
  ```

---

## Running the Chatbot

You can choose between a CLI version or a modern Web UI via Streamlit.

### Option A: Streamlit Web UI (Recommended)
```bash
streamlit run streamlit_app.py
```
This will open the interface in your browser at `http://localhost:8501`.

### Option B: CLI Version
```bash
python3 app.py
```
- Type `/switch` to change personalities.
- Type `/exit` to quit.

3. **How the RAG Pipeline Works**:
   - The user inputs a message. The `sentence-transformers` model creates a 384d vector representing its semantic meaning.
   - This vector is passed to Endee through `/api/v1/index/chat_memory/search` to retrieve the Top-3 past dialogue snippets.
   - The retrieved context + the current personality (mentor, friend, etc.) + the original input is sent to the LLM via Groq.
   - Following the LLM's response, the new (user_query, bot_response) pair is embedded and saved entirely in Endee (`/api/v1/index/chat_memory/vector/insert`) for future context!
   
4. **Performance & Improvement Suggestions**:
   - **Sparse + Dense (Hybrid Search)**: Endee supports sparse vectors. We could use BM25 to combine exact-keyword and semantic matches for better accuracy.
   - **Chunking Strategy**: For very long replies, we could chunk the metadata for better search granularity.
   - **Prefiltering**: We can assign "user_id" or "chat_session" inside Endee's metadata/filters to strictly divide contexts.