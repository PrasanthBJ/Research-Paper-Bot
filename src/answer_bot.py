
from src.retrieval import search
from src.generator import generate_answer


def ask_question(question):
    print("Searching research papers...", flush=True)

    retrieved_chunks = search(
        query=question,
        top_k=3
    )

    if not retrieved_chunks:
        return {
            "answer": "No relevant passages were found.",
            "sources": []
        }

    print("Generating answer...", flush=True)

    answer = generate_answer(
        question,
        retrieved_chunks
    )

    return {
        "answer": answer,
        "sources": retrieved_chunks
    }

if __name__ == "__main__":
    print("Research Paper Answer Bot")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nAsk a question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        try:
            answer = ask_question(question)

            print("\nAnswer:\n")
            print(answer)

        except Exception as error:
            print(f"\nError: {error}")