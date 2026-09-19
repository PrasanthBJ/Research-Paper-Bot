
import streamlit as st

from src.answer_bot import ask_question


st.set_page_config(
    page_title="Research Paper Answer Bot",
    page_icon="📚",
    layout="wide"
)

st.title("Research Paper Answer Bot")

st.write(
    "Ask questions about your research papers "
    "and explore the supporting passages."
)

question = st.text_input(
    "Enter your research question",
    placeholder="How does the Transformer use attention?"
)

if st.button("Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner(
            "Searching papers and generating answer..."
        ):
            try:
                result = ask_question(question)

                st.subheader("Generated Answer")
                st.write(result["answer"])

                st.subheader("Retrieved Sources")

                for i, source in enumerate(
                    result["sources"],
                    start=1
                ):
                    metadata = source["metadata"]

                    with st.expander(
                        f"Source {i}: "
                        f"{metadata['title']} "
                        f"(Page {metadata['page']})"
                    ):
                        st.write(source["text"])

                        st.caption(
                            f"Similarity: "
                            f"{source['score']:.4f}"
                        )

            except Exception as error:
                st.error(str(error))