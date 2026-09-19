
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from src.ingest import load_research_papers


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for document in documents:
        text_chunks = splitter.split_text(
            document["text"]
        )

        for chunk_index, chunk_text in enumerate(
            text_chunks
        ):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **document["metadata"],
                    "chunk_index": chunk_index
                }
            })

    return chunks


if __name__ == "__main__":
    documents = load_research_papers()
    chunks = create_chunks(documents)

    print(f"Total pages: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

    if chunks:
        print("\nFirst chunk metadata:")
        print(chunks[0]["metadata"])

        print("\nFirst chunk text:")
        print(chunks[0]["text"][:500])