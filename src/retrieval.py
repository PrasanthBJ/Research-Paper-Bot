    
from pathlib import Path
import json

import faiss
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_ID = "minilm"

INDEX_PATH = Path("models") / f"{MODEL_ID}.faiss"
METADATA_PATH = Path("models") / f"{MODEL_ID}_metadata.json"


# Load once, not for every question
model = None
index = None
chunks = None


def load_retriever():
    global model, index, chunks

    if model is not None:
        return

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_PATH}\n"
            "Run the embedding pipeline first."
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    print("Loading embedding model...", flush=True)
    model = SentenceTransformer(MODEL_NAME)

    print("Loading FAISS index...", flush=True)
    index = faiss.read_index(str(INDEX_PATH))

    with open(METADATA_PATH, encoding="utf-8") as file:
        chunks = json.load(file)


def search(query, top_k=3):
    load_retriever()

    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    scores, indices = index.search(
        query_vector.astype("float32"),
        min(top_k, len(chunks))
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        chunk = chunks[int(idx)]

        results.append({
            "score": float(score),
            "text": chunk["text"],
            "metadata": chunk["metadata"]
        })

    return results


if __name__ == "__main__":
    question = input("Enter a research question: ")

    results = search(question)

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['metadata']['title']}")
        print(f"Page: {result['metadata']['page']}")
        print(result["text"][:500])