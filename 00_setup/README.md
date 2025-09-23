00) Setup — Environment & Install (with Ollama)

Story (short): Nova found a big friendly robot brain. LangChain is the helpful organizer that shows Nova how to write good letters (prompts), line up steps (chains), remember chats (memory), and use gadgets (tools and agents). Today, Nova makes the workshop ready and says “hello” to a local brain using Ollama.

What we’ll do
- Use a single Python virtual environment at the repo root (recommended) or just install locally
- Install LangChain and community integrations
- (Optional) Prepare a .env file (useful for cloud keys later). Ollama needs no key
- Run a tiny hello-world chain with a local Ollama model

Prerequisites
- Python 3.9+
- Ollama installed and running (start the background service)
  - Pull a model, e.g.: `ollama pull llama3` (or your preferred model)

Install (one-time)
```bash
# from repo root (recommended single venv)
cd ~/Downloads/langchain-a2z
python3 -m venv .venv
source .venv/bin/activate

# install minimal deps for local Ollama + LangChain
pip install --upgrade pip
pip install -r 00_setup/requirements.txt
```

Environment variables (.env)
- Not required for Ollama.
- If you later use OpenAI or LangSmith, copy `.env.example` to `.env` at repository root and fill values.

Hello world (Ollama)
```bash
source ~/Downloads/langchain-a2z/.venv/bin/activate
python 00_setup/hello_world_ollama.py
```

Expected output
- A friendly response like: "Hello, Nova!" from your local model.

Notes
- If you see model errors, ensure you’ve pulled a model (e.g., `ollama pull llama3`).
- You can swap `model_name` in the script to any model you have locally.


