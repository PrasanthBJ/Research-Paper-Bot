
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunking import load_research_papers, create_chunks


# Paths
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

INDEX_PATH = MODELS_DIR / "minilm.faiss"
METADATA_PATH = MODELS_DIR / "minilm_metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_index():
    print("Loading research papers...", flush=True)

    documents = load_research_papers()
    chunks = create_chunks(documents)

    if not chunks:
        raise ValueError("No chunks found!")

    print(f"Total chunks: {len(chunks)}", flush=True)

    # Extract text from chunks
    texts = [chunk["text"] for chunk in chunks]

    print("Loading embedding model...", flush=True)

    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...", flush=True)

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    print("Building FAISS index...", flush=True)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print("Saving FAISS index...", flush=True)

    faiss.write_index(index, str(INDEX_PATH))

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\nIndex created successfully!")
    print(f"Vectors: {index.ntotal}")
    print(f"Dimension: {dimension}")
    print(f"Index saved: {INDEX_PATH}")
    print(f"Metadata saved: {METADATA_PATH}")


if __name__ == "__main__":
    build_index()