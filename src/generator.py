
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_answer(question, retrieved_chunks):
    context = "\n\n".join(
        chunk["text"]
        for chunk in retrieved_chunks
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research paper assistant. "
                    "Answer using only the supplied context. "
                    "If the context does not contain the answer, "
                    "say so. Do not invent citations."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ],
        max_completion_tokens=400,
        temperature=0.2
    )

    return response.choices[0].message.content