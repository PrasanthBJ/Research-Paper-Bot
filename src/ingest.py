
from pathlib import Path
from pypdf import PdfReader

DATA_DIR = Path("data/papers")


def load_research_papers(data_dir=DATA_DIR):
    documents = []

    pdf_files = sorted(data_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {data_dir.resolve()}"
        )

    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")

        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if not text:
                continue

            documents.append({
                "text": text,
                "metadata": {
                    "source": pdf_path.name,
                    "title": pdf_path.stem,
                    "page": page_number
                }
            })

    return documents


if __name__ == "__main__":
    docs = load_research_papers()

    print(f"\nExtracted pages: {len(docs)}")

    if docs:
        print("\nFirst page metadata:")
        print(docs[0]["metadata"])

        print("\nExtracted text:")
        print(docs[0]["text"][:500])