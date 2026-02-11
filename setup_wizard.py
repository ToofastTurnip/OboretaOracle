"""Oboreta Oracle — first-run setup wizard.

Usage:
    python setup_wizard.py
"""

import shutil
import subprocess
import sys


MODELS = ["gemma3:12b", "nomic-embed-text"]


def check_python():
    v = sys.version_info
    print(f"[1/4] Python version: {v.major}.{v.minor}.{v.micro}")
    if v < (3, 10):
        print("  ✗ Python 3.10+ is required.")
        raise SystemExit(1)
    print("  ✓ OK")


def install_dependencies():
    print("[2/4] Installing Python dependencies…")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=True,
    )
    print("  ✓ Dependencies installed")


def check_ollama():
    print("[3/4] Checking for Ollama…")
    if shutil.which("ollama") is None:
        print("  ✗ Ollama not found on PATH.")
        print("    Install it from https://ollama.com and re-run this script.")
        raise SystemExit(1)
    print("  ✓ Ollama found")


def pull_models():
    print("[4/4] Pulling Ollama models (this may take a while)…")
    for model in MODELS:
        print(f"  → ollama pull {model}")
        subprocess.run(["ollama", "pull", model], check=True)
        print(f"  ✓ {model} ready")


def main():
    print("=" * 50)
    print("  🐉 Oboreta Oracle — Setup Wizard")
    print("=" * 50)
    print()

    check_python()
    install_dependencies()
    check_ollama()
    pull_models()

    print()
    print("=" * 50)
    print("  Setup complete! To launch the app, run:")
    print()
    print("    streamlit run app.py")
    print()
    print("  On the sidebar, point it at a folder of")
    print("  .txt, .md, or .pdf notes and hit")
    print("  'Build / Update Database' to get started.")
    print("=" * 50)


if __name__ == "__main__":
    main()
