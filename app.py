import os

import streamlit as st

from rag_pipeline import generate_rag_answer, initialize_vector_store, page_label

st.set_page_config(page_title="RAG Chatbot", layout="centered")

st.title("📚 Document RAG Chatbot with Gemini")
st.markdown("Upload a PDF document and chat intelligently with its contents.")

# Streamlit re-runs this whole script on every interaction, so anything that must
# survive between runs lives in st.session_state.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_file" not in st.session_state:
    st.session_state.indexed_file = None


def show_sources(docs):
    """Show the retrieved chunks with human-friendly page numbers (page 1 = first page)."""
    with st.expander("View Source Document Chunks"):
        for i, doc in enumerate(docs, start=1):
            st.markdown(f"**Chunk {i} ({page_label(doc)})**")
            st.text(doc.page_content)  # plain text, so symbols in the PDF can't break the layout


# ---- Sidebar: upload and process a document ----
st.sidebar.header("1. Upload Knowledge Base")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None and st.sidebar.button("Process Document"):
    os.makedirs("data", exist_ok=True)
    file_name = os.path.basename(uploaded_file.name)  # never trust a file name from a browser
    file_path = os.path.join("data", file_name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing chunks and building vector database..."):
        try:
            initialize_vector_store(file_path)
        except Exception as error:
            st.sidebar.error(f"Could not process this PDF: {error}")
        else:
            st.session_state.indexed_file = file_name
            st.session_state.messages = []  # new document, so start a fresh chat
            st.sidebar.success("Document processed successfully!")

if st.session_state.indexed_file:
    st.sidebar.caption(f"Active document: {st.session_state.indexed_file}")

# ---- Main chat interface ----
st.header("2. Chat with your Document")
st.caption("Each question is answered on its own from the document; earlier messages are shown for reference only.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

question = st.chat_input("Ask a question about your document")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking and generating answer..."):
            answer, relevant_docs = generate_rag_answer(question)

        if answer is None:  # nothing has been indexed yet
            answer = "Please upload and process a document using the sidebar first."
            relevant_docs = []

        st.markdown(answer)
        if relevant_docs:
            show_sources(relevant_docs)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": relevant_docs}
    )