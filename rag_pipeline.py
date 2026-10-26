import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# ---- Settings you can tune (and record the results in your evaluation table) ----
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # change in .env if the name changes
CHUNK_SIZE = 500      # characters, not words or tokens
CHUNK_OVERLAP = 50
TOP_K = 5
USE_MMR = False       # True = pick diverse chunks instead of near-duplicates (an experiment to try)
TEMPERATURE = 0       # 0 gives repeatable answers, which you need for fair evaluation


@lru_cache(maxsize=1)
def get_embeddings():
    """Load the embedding model once. Loading it on every question is slow."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_llm():
    """Create the Gemini client once."""
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=TEMPERATURE)


def initialize_vector_store(pdf_path):
    """Loads a PDF, splits it into chunks, and stores them in ChromaDB.

    Any previously indexed document is replaced, so chunks from different PDFs
    (or the same PDF processed twice) never pile up together.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Could not find file at: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    raw_documents = loader.load()

    # Scanned PDFs have no extractable text; fail with a clear message instead of a crash later
    if not any(doc.page_content.strip() for doc in raw_documents):
        raise ValueError(
            "No text could be extracted from this PDF. It may be a scanned image; "
            "try a text-based PDF."
        )

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = text_splitter.split_documents(raw_documents)

    embeddings = get_embeddings()

    # Remove the old index first (works on Windows too, unlike deleting the folder)
    try:
        Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
        ).delete_collection()
    except Exception:
        pass  # nothing to delete on the first run

    vector_store = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )
    return vector_store


def retrieve(user_query, k=TOP_K):
    """Return the k most relevant chunks. Kept separate so you can evaluate retrieval alone."""
    if not os.path.exists(PERSIST_DIRECTORY):
        return []

    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=get_embeddings(),
    )
    if USE_MMR:
        retriever = vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": k, "fetch_k": k * 4}
        )
    else:
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(user_query)


def page_label(doc):
    """PyPDFLoader counts pages from 0, but people count from 1."""
    page = doc.metadata.get("page")
    return f"page {page + 1}" if isinstance(page, int) else "page ?"


def generate_rag_answer(user_query):
    """Retrieves relevant chunks and passes them to Gemini for an answer with citations.

    Returns (answer_text, relevant_docs). Returns (None, []) if nothing is indexed yet.
    """
    if not os.path.exists(PERSIST_DIRECTORY):
        return None, []

    relevant_docs = retrieve(user_query)
    if not relevant_docs:
        return "I could not find this in the document.", []

    # Number each chunk so the model can cite it, e.g. [2]
    context_text = "\n\n".join(
        f"[{i}] ({page_label(doc)})\n{doc.page_content}"
        for i, doc in enumerate(relevant_docs, start=1)
    )

    prompt = (
        "You are an expert assistant for question-answering tasks. "
        "Use ONLY the numbered context passages below to answer the user's question accurately. "
        "Cite the passages you used like [1] or [2]. "
        "If the answer cannot be found in the context, reply exactly: "
        "I could not find this in the document.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {user_query}\n"
        "Answer:"
    )

    try:
        response = get_llm().invoke(prompt)
    except Exception as error:  # bad key, quota, network... don't crash the app
        message = (
            f"Sorry, the language model request failed ({type(error).__name__}). "
            "Check your GOOGLE_API_KEY, your quota and your internet connection."
        )
        return message, relevant_docs

    content = response.content
    if isinstance(content, list):  # some model versions return a list of text blocks
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return content, relevant_docs