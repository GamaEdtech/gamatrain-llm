import uvicorn

# When executed as a script (python api/llm_server_modular.py),
# import via the local package structure.
from app.main import app


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

