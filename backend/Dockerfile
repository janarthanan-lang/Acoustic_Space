FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fine-tunes the AST model at build time so the image is ready to serve.
# Requires network access to huggingface.co during `docker build` (downloads
# the ~340MB base checkpoint). If your build environment can't reach the
# Hub, comment this out and instead COPY a pre-trained weights/ast_finetuned/
# directory into the image, or let main.py fall back to the untrained base
# head at container startup (see model_ast.py's load_model()).
RUN python train_ast_model.py

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
