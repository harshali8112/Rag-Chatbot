Document Q&A Chatbot (RAG)

A chatbot that answers questions from your PDF files and shows the source of each answer.

Live demo: [add link] | Tech: Python, LangChain, ChromaDB, Streamlit

How it works

PDF → chunks → embeddings → ChromaDB → retrieve relevant chunks → LLM answers using only those chunks.
