"""Oboreta Oracle — Streamlit chat interface for D&D notes RAG."""

from pathlib import Path

import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

from utils.config import load_config, update_config
from ingest import build_vector_store

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Oboreta Oracle", page_icon="🐉", layout="wide")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🐉 Dungeon Archive")
    st.caption("Your local D&D notes, searchable.")

    config = load_config()

    # -- Source selector -----------------------------------------------------
    source_type = st.radio(
        "Choose source",
        options=["Local Folder", "Google Drive"],
        index=0 if config["source_type"] == "local" else 1,
    )

    if source_type == "Local Folder":
        source_path = st.text_input(
            "Absolute folder path",
            value=config["local_folder_path"],
            placeholder="/home/user/dnd-notes",
        )
    else:
        source_path = st.text_input(
            "Google Drive Folder ID",
            value=config["google_drive_folder_id"],
            placeholder="1aBcDeFgHiJkLmNoPqRsTuVwXyZ",
        )

    # -- Build / Update button -----------------------------------------------
    if st.button("Build / Update Database", use_container_width=True):
        if not source_path:
            st.error("Please enter a source path or folder ID first.")
        else:
            src_key = "local" if source_type == "Local Folder" else "drive"
            # Persist the user's choice so it survives restarts.
            update_config(
                source_type=src_key,
                local_folder_path=source_path if src_key == "local" else config["local_folder_path"],
                google_drive_folder_id=source_path if src_key == "drive" else config["google_drive_folder_id"],
            )

            with st.spinner("Indexing documents — this may take a moment…"):
                try:
                    result = build_vector_store(source_path, src_key)
                    st.success(result)
                    # Force the QA chain to reload with the fresh vector store.
                    st.session_state.qa_chain = None
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    # -- Database status -----------------------------------------------------
    st.divider()
    db_path = Path(config["chroma_db_path"]).resolve()
    if db_path.exists() and any(db_path.iterdir()):
        st.info("Vector database found — ready to chat.")
    else:
        st.warning("No database yet. Index some documents first.")


# ---------------------------------------------------------------------------
# Build (or re-use) the RetrievalQA chain
# ---------------------------------------------------------------------------
def get_qa_chain():
    """Return a RetrievalQA chain backed by the persistent ChromaDB."""
    if st.session_state.qa_chain is not None:
        return st.session_state.qa_chain

    config = load_config()
    db_path = str(Path(config["chroma_db_path"]).resolve())

    if not Path(db_path).exists() or not any(Path(db_path).iterdir()):
        return None

    embeddings = OllamaEmbeddings(
        model=config["embedding_model"],
        base_url=config["ollama_base_url"],
    )

    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name="dnd_notes",
    )

    llm = ChatOllama(
        model=config["llm_model"],
        base_url=config["ollama_base_url"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
    )

    st.session_state.qa_chain = chain
    return chain


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("🐉 Oboreta Oracle")

# Render conversation history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                st.markdown(msg["sources"])

# Chat input.
if prompt := st.chat_input("Ask a question about your notes…"):
    # Show user message immediately.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response.
    chain = get_qa_chain()

    if chain is None:
        reply = "I don't have any notes indexed yet. Use the sidebar to build the database first."
        sources_md = ""
    else:
        with st.spinner("Thinking…"):
            result = chain.invoke({"query": prompt})
        reply = result["result"]

        # Format source documents for the expander.
        seen = set()
        source_lines = []
        for doc in result.get("source_documents", []):
            src = doc.metadata.get("source", "unknown")
            if src not in seen:
                seen.add(src)
                page = doc.metadata.get("page")
                label = f"`{src}`" + (f" — page {page}" if page is not None else "")
                snippet = doc.page_content[:200].replace("\n", " ")
                source_lines.append(f"- {label}\n  > {snippet}…")
        sources_md = "\n".join(source_lines) if source_lines else ""

    # Display assistant message.
    with st.chat_message("assistant"):
        st.markdown(reply)
        if sources_md:
            with st.expander("Sources"):
                st.markdown(sources_md)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources_md}
    )
