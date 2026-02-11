import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.json"

DEFAULTS = {
    "source_type": "local",
    "local_folder_path": "",
    "google_drive_folder_id": "",
    "chroma_db_path": "chroma_db",
    "chunk_size": 2000,
    "chunk_overlap": 300,
    "ollama_base_url": "http://localhost:11434",
    "llm_model": "gemma3:12b",
    "embedding_model": "nomic-embed-text",
}


def load_config() -> dict:
    """Load config from disk, filling in any missing keys with defaults."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            saved = json.load(f)
        return {**DEFAULTS, **saved}
    return DEFAULTS.copy()


def save_config(config: dict) -> None:
    """Write the current config dict to disk."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)


def update_config(**kwargs) -> dict:
    """Load config, merge in any keyword args, save, and return the result."""
    config = load_config()
    config.update(kwargs)
    save_config(config)
    return config
