"""Oboreta Oracle — Streamlit chat interface for D&D notes RAG."""

import shutil
from pathlib import Path

import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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
if "local_folder_path" not in st.session_state:
    st.session_state.local_folder_path = ""

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🐉 Oboreta Oracle")
    st.caption("Your local D&D notes, searchable.")

    config = load_config()

    # -- Source selector -----------------------------------------------------
    source_type = st.radio(
        "Choose source",
        options=["Local Folder", "Google Drive"],
        index=0 if config["source_type"] == "local" else 1,
    )

    if source_type == "Local Folder":
        # Initialize from saved config on first load.
        if not st.session_state.local_folder_path:
            st.session_state.local_folder_path = config["local_folder_path"]

        if st.button("Browse…", use_container_width=True):
            import subprocess

            result = subprocess.run(
                [
                    "osascript", "-e",
                    'POSIX path of (choose folder with prompt "Select your notes folder")',
                ],
                capture_output=True,
                text=True,
            )
            folder = result.stdout.strip()
            if folder:
                st.session_state.local_folder_path = folder

        source_path = st.text_input(
            "Folder path",
            value=st.session_state.local_folder_path,
            placeholder="/home/user/dnd-notes",
            key="folder_input",
        )
        # Keep session state in sync if the user edits the text field directly.
        st.session_state.local_folder_path = source_path
    else:
        creds_path = Path("config/credentials.json")
        has_creds = creds_path.exists()

        if has_creds:
            st.success("Google Drive credentials installed.", icon="✅")
            source_path = st.text_input(
                "Google Drive Folder ID",
                value=config["google_drive_folder_id"],
                placeholder="1aBcDeFgHiJkLmNoPqRsTuVwXyZ",
                help="The ID is the last segment of the folder URL.",
            )
        else:
            source_path = ""
            with st.expander("Set up Google Drive connection", expanded=True):
                st.markdown(
                    "**To connect Google Drive you need an OAuth credentials file.**\n\n"
                    "1. Go to the [Google Cloud Console]"
                    "(https://console.cloud.google.com/).\n"
                    "2. Create a project (or use an existing one).\n"
                    "3. Enable the **Google Drive API**.\n"
                    "4. Go to **Credentials** > **Create Credentials** "
                    "> **OAuth client ID**.\n"
                    "   - Application type: **Desktop app**.\n"
                    "   - Download the JSON file.\n"
                    "5. Upload the file below."
                )
                uploaded = st.file_uploader(
                    "Upload credentials JSON",
                    type=["json"],
                    key="gdrive_creds_upload",
                )
                if uploaded is not None:
                    creds_path.parent.mkdir(parents=True, exist_ok=True)
                    creds_path.write_bytes(uploaded.getvalue())
                    st.success("Credentials saved! Reloading…")
                    st.rerun()

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
# Prompts
# ---------------------------------------------------------------------------
MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    "You are an assistant that helps retrieve information from D&D campaign notes. "
    "The user asked the following question:\n\n"
    "{question}\n\n"
    "Generate 3 alternative versions of this question that use different words "
    "and synonyms to help find relevant information. Think about related concepts — "
    "for example, 'dating' could also mean 'romantic partner', 'in a relationship', "
    "'lover', 'kissing', etc. 'Adversaries' could mean 'enemies', 'attackers', "
    "'threats', etc.\n\n"
    "Return ONLY the 3 alternative questions, one per line, with no numbering or bullets."
)

RAG_PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using ONLY the context below. "
    "Be short and direct — 1 to 3 sentences unless a list is genuinely needed. "
    "Do not speculate, infer, or add information not explicitly stated in the context. "
    "Do not explain your reasoning. "
    "If the context does not contain the answer, say: "
    "'I don't have enough information in my notes to answer that.'\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def _format_docs(docs):
    """Join retrieved document pages into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def _interleave_and_dedupe(doc_lists, max_docs=8):
    """Round-robin interleave from multiple result lists, deduping by content.

    This ensures each query's unique results get fair representation
    rather than one query dominating the context.
    """
    seen = set()
    result = []
    max_len = max((len(dl) for dl in doc_lists), default=0)
    for i in range(max_len):
        if len(result) >= max_docs:
            break
        for dl in doc_lists:
            if len(result) >= max_docs:
                break
            if i < len(dl):
                doc = dl[i]
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    result.append(doc)
    return result


# ---------------------------------------------------------------------------
# Build (or re-use) the retrieval chain components
# ---------------------------------------------------------------------------
def get_qa_chain():
    """Return (llm, retriever, multi_query_chain) backed by the persistent ChromaDB."""
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

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatOllama(
        model=config["llm_model"],
        base_url=config["ollama_base_url"],
    )

    # Chain that generates alternative query phrasings.
    multi_query_chain = MULTI_QUERY_PROMPT | llm | StrOutputParser()

    result = (llm, retriever, multi_query_chain)
    st.session_state.qa_chain = result
    return result


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

# Show a welcome state when there's no conversation yet.
if not st.session_state.messages:
    st.markdown(
        "<div style='text-align:center; padding-top:4rem; color:#888;'>"
        "<p style='font-size:2.5rem; margin-bottom:0.25rem;'>🐉</p>"
        "<p style='font-size:1.1rem;'>Ask a question about your campaign notes to get started.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

# Render conversation history.
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                st.markdown(msg["sources"])

# Chat input.
if prompt := st.chat_input("Ask a question about your notes…"):
    # Show user message immediately.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response inside the assistant bubble.
    with st.chat_message("assistant"):
        qa = get_qa_chain()

        if qa is None:
            reply = "I don't have any notes indexed yet. Use the sidebar to build the database first."
            sources_md = ""
        else:
            llm, retriever, multi_query_chain = qa
            with st.spinner("Thinking…"):
                # 1. Generate alternative phrasings of the question.
                alt_text = multi_query_chain.invoke({"question": prompt})
                alt_queries = [q.strip() for q in alt_text.strip().splitlines() if q.strip()]

                # 2. Retrieve chunks for the original + all alternative queries.
                doc_lists = [retriever.invoke(prompt)]
                for alt_q in alt_queries:
                    doc_lists.append(retriever.invoke(alt_q))

                # 3. Interleave round-robin so each query gets fair representation.
                source_docs = _interleave_and_dedupe(doc_lists, max_docs=8)
                context = _format_docs(source_docs)

                # 4. Generate the final answer.
                answer_chain = RAG_PROMPT | llm | StrOutputParser()
                reply = answer_chain.invoke({"context": context, "question": prompt})

            # Format source documents for the expander.
            seen = set()
            source_lines = []
            for doc in source_docs:
                src = doc.metadata.get("source", "unknown")
                if src not in seen:
                    seen.add(src)
                    page = doc.metadata.get("page")
                    label = f"`{src}`" + (f" — page {page}" if page is not None else "")
                    snippet = doc.page_content[:200].replace("\n", " ")
                    source_lines.append(f"- {label}\n  > {snippet}…")
            sources_md = "\n".join(source_lines) if source_lines else ""

        st.markdown(reply)
        if sources_md:
            with st.expander("Sources", expanded=False):
                st.markdown(sources_md)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources_md}
    )

    # Play a soft pop sound to notify the user the answer is ready.
    st.components.v1.html(
        """
        <script>
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const o = ctx.createOscillator();
        const g = ctx.createGain();
        o.type = 'sine';
        o.frequency.setValueAtTime(880, ctx.currentTime);
        o.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.12);
        g.gain.setValueAtTime(0.3, ctx.currentTime);
        g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.2);
        o.connect(g);
        g.connect(ctx.destination);
        o.start();
        o.stop(ctx.currentTime + 0.2);
        </script>
        """,
        height=0,
    )
