import os

# Fast, deterministic CI/local runs without embedding downloads or index builds.
os.environ.setdefault("ENABLE_RAG", "false")
