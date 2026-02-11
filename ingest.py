"""Document loading, chunking, and ChromaDB ingestion pipeline."""

from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from utils.config import load_config

# ---------------------------------------------------------------------------
# Loaders keyed by file extension
# ---------------------------------------------------------------------------
LOADER_MAP = {
    ".txt": (TextLoader, {"encoding": "utf-8"}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".pdf": (PyPDFLoader, {}),
}


# ---------------------------------------------------------------------------
# Local folder loader
# ---------------------------------------------------------------------------
def _load_local(source_path: str) -> list:
    """Recursively load .txt, .md, and .pdf files from a local directory."""
    folder = Path(source_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {source_path}")

    documents = []
    for ext, (loader_cls, loader_kwargs) in LOADER_MAP.items():
        loader = DirectoryLoader(
            str(folder),
            glob=f"**/*{ext}",
            loader_cls=loader_cls,
            loader_kwargs=loader_kwargs,
            show_progress=True,
            use_multithreading=True,
        )
        docs = loader.load()
        print(f"  Loaded {len(docs)} page(s) from {ext} files")
        documents.extend(docs)

    return documents


# ---------------------------------------------------------------------------
# Google Drive loader — uses utils/drive_auth for OAuth
# ---------------------------------------------------------------------------
def _load_drive(folder_id: str) -> list:
    """Load documents from a Google Drive folder.

    Authenticates via utils.drive_auth (browser-based OAuth on first run,
    cached token thereafter).  See utils/drive_auth.py for setup steps.
    """
    try:
        from langchain_google_community import GoogleDriveLoader
    except ImportError:
        raise ImportError(
            "Install langchain-google-community to use Google Drive: "
            "pip install 'langchain-google-community[drive]'"
        )

    from utils.drive_auth import authenticate

    creds = authenticate()

    loader = GoogleDriveLoader(
        folder_id=folder_id,
        recursive=True,
        file_types=["document", "pdf"],
        credentials=creds,
    )
    docs = loader.load()
    print(f"  Loaded {len(docs)} document(s) from Google Drive")
    return docs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_vector_store(
    source_path: str,
    source_type: str = "local",
) -> str:
    """Load documents, split them into chunks, embed, and persist to ChromaDB.

    Args:
        source_path: Local directory path or Google Drive folder ID.
        source_type: "local" or "drive".

    Returns:
        A status message summarising what was indexed.
    """
    config = load_config()

    # ---- 1. Load raw documents ----
    print(f"[1/4] Loading documents ({source_type})…")
    if source_type == "local":
        documents = _load_local(source_path)
    elif source_type == "drive":
        documents = _load_drive(source_path)
    else:
        raise ValueError(f"Unknown source_type: {source_type!r}")

    if not documents:
        return "No documents found. Check the source path and try again."

    print(f"  Total raw pages/documents: {len(documents)}")

    # ---- 1b. Inject directory path context into each document ----
    # This ensures folder names (e.g. "Sunbell/blacksmith.txt") become part
    # of the searchable text, so the retriever can connect documents to their
    # parent directories even when the content doesn't mention them.
    root = Path(source_path).resolve() if source_type == "local" else None
    for doc in documents:
        src = doc.metadata.get("source", "")
        if root and src:
            rel = Path(src).resolve().relative_to(root)
            doc.page_content = f"[Source: {rel}]\n\n{doc.page_content}"
        elif src:
            doc.page_content = f"[Source: {src}]\n\n{doc.page_content}"

    # ---- 2. Split into chunks ----
    print("[2/4] Splitting into chunks…")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )
    chunks = splitter.split_documents(documents)
    print(f"  Created {len(chunks)} chunks")

    # ---- 3. Create embeddings + persist to ChromaDB ----
    print("[3/4] Embedding chunks (this may take a while on first run)…")
    embeddings = OllamaEmbeddings(
        model=config["embedding_model"],
        base_url=config["ollama_base_url"],
    )

    db_path = str(Path(config["chroma_db_path"]).resolve())
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name="dnd_notes",
    )

    # ---- 4. Done ----
    count = vectorstore._collection.count()
    msg = (
        f"[4/4] Done! Indexed {len(chunks)} chunks "
        f"({len(documents)} source pages) into ChromaDB "
        f"({count} vectors stored at {db_path})."
    )
    print(msg)
    return msg


# ---------------------------------------------------------------------------
# CLI convenience — run directly to ingest from config settings
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = load_config()
    src_type = cfg["source_type"]
    src_path = (
        cfg["local_folder_path"]
        if src_type == "local"
        else cfg["google_drive_folder_id"]
    )

    if not src_path:
        print(
            "Error: No source path configured. "
            "Set local_folder_path or google_drive_folder_id in config/config.json."
        )
        raise SystemExit(1)

    result = build_vector_store(src_path, src_type)
    print(f"\n{result}")
