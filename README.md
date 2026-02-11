# Oboreta Oracle

A local RAG (Retrieval Augmented Generation) app for chatting with your Dungeons & Dragons notes. Everything runs on your machine — no API keys, no cloud LLMs, no data leaving your computer.

Point it at a folder of `.txt`, `.md`, or `.pdf` files (or a Google Drive folder), and it builds a searchable vector database you can query through a chat interface. Every answer includes the source documents it was drawn from.

## How It Works

```
Your Notes (.txt, .md, .pdf)
        │
        ▼
   ┌──────────┐    nomic-embed-text     ┌──────────┐
   │  ingest   │ ─────────────────────► │ ChromaDB │
   │  pipeline │    split + embed        │ (vectors)│
   └──────────┘                         └────┬─────┘
                                             │
   User Question                             │ retrieve top-5 chunks
        │                                    │
        ▼                                    ▼
   ┌──────────┐                        ┌──────────┐
   │ Streamlit │ ◄──────────────────── │ gemma3   │
   │    UI     │    generated answer   │  :12b    │
   └──────────┘                        └──────────┘
```

1. **Ingestion** — Documents are loaded, split into 2000-character chunks (with 300-char overlap), embedded with `nomic-embed-text` via Ollama, and stored in a persistent ChromaDB database.
2. **Retrieval** — When you ask a question, the app embeds your query, finds the 5 most relevant chunks from the database, and passes them as context to the LLM.
3. **Generation** — `gemma3:12b` reads the retrieved chunks and your question, then generates a grounded answer. Source documents are shown in a collapsible panel beneath each response.

## Prerequisites

- **Python 3.10+**
- **Ollama** — Install from [ollama.com](https://ollama.com)

## Quick Start

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd OboretaOracle

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

# 3. Run the setup wizard
python setup_wizard.py
```

The setup wizard will:
- Verify your Python version
- Install all dependencies from `requirements.txt`
- Check that Ollama is installed
- Pull `gemma3:12b` and `nomic-embed-text`

Once setup completes:

```bash
# 4. Launch the app
streamlit run app.py
```

## Usage

### Indexing Local Notes

1. Open the app in your browser (Streamlit will print the URL).
2. In the sidebar under **Dungeon Archive**, select **Local Folder**.
3. Paste the absolute path to your notes folder (e.g. `/home/user/dnd-notes`).
4. Click **Build / Update Database** and wait for indexing to finish.
5. Start chatting.

### Indexing Google Drive Notes

Google Drive requires a one-time OAuth setup:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a project (or use an existing one).
3. Enable the **Google Drive API**.
4. Go to **Credentials** > **Create Credentials** > **OAuth client ID**.
   - Application type: **Desktop app**.
   - Download the JSON file.
5. Rename it to `credentials.json` and place it at `config/credentials.json`.
6. Install the optional Drive dependency:
   ```bash
   pip install langchain-google-community[drive]
   ```
7. In the app sidebar, select **Google Drive**, paste your folder ID, and click **Build / Update Database**.
8. On first run, a browser window will open for you to authorize access. A `token.json` is saved at `config/token.json` so you won't need to authorize again.

> The folder ID is the last segment of the Google Drive folder URL:
> `https://drive.google.com/drive/folders/`**`1aBcDeFgHiJkLmNoPqRsTuVwXyZ`**

### Chatting

Type a question into the chat input at the bottom of the page. The Oracle will retrieve relevant chunks from your notes and generate an answer. Click the **Sources** expander beneath any response to see exactly which documents (and pages) were used.

### Re-indexing

If you add, edit, or remove notes, click **Build / Update Database** again. The vector store will be rebuilt from scratch with the current contents of your source folder.

## Project Structure

```
OboretaOracle/
├── setup_wizard.py        # First-run setup: deps, Ollama, model pulls
├── app.py                 # Streamlit UI — sidebar + chat interface
├── ingest.py              # Document loading, chunking, embedding, storage
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (Ollama URL, etc.)
├── .gitignore
├── config/
│   └── config.json        # Persisted user settings
├── chroma_db/             # Vector database (created on first indexing)
├── data/                  # Default local notes folder (optional)
└── utils/
    ├── __init__.py
    ├── config.py           # Load / save / update config helpers
    └── drive_auth.py       # Google Drive OAuth flow
```

## Configuration

All settings live in `config/config.json` and are editable through the UI or by hand:

| Key | Default | Description |
|---|---|---|
| `source_type` | `local` | `local` or `drive` |
| `local_folder_path` | | Absolute path to local notes |
| `google_drive_folder_id` | | Google Drive folder ID |
| `chroma_db_path` | `chroma_db` | Where ChromaDB stores vectors |
| `chunk_size` | `2000` | Characters per text chunk |
| `chunk_overlap` | `300` | Overlap between adjacent chunks |
| `ollama_base_url` | `http://localhost:11434` | Ollama server URL |
| `llm_model` | `gemma3:12b` | Chat model |
| `embedding_model` | `nomic-embed-text` | Embedding model |

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Ollama — gemma3:12b |
| Embeddings | Ollama — nomic-embed-text |
| Vector Store | ChromaDB |
| Orchestration | LangChain |
| Language | Python 3.10+ |

## Troubleshooting

**Ollama connection refused** — Make sure the Ollama server is running. Start it with `ollama serve` or launch the Ollama desktop app.

**Ingestion is slow** — Embedding thousands of chunks takes time on the first run. Subsequent queries are fast since the vectors are persisted.

**Google Drive "credentials not found"** — Ensure `config/credentials.json` exists and is a valid OAuth Desktop-app JSON from Google Cloud Console.

**Model not found** — Run `ollama list` to see installed models. If a model is missing, pull it with `ollama pull <model-name>`.
