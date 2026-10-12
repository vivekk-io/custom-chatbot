# Local Privacy-First Document RAG Assistant

An end-to-end Retrieval-Augmented Generation (RAG) web application running completely offline on Apple Silicon, eliminating cloud API costs and ensuring local data privacy.

## Features
- **Multi-Source Ingestion:** Supports local PDF documents (`PyPDFLoader`) and static web pages (`WebBaseLoader`).
- **Context-Aware Chunking:** Uses `RecursiveCharacterTextSplitter` (1000 chunk size, 150 overlap) to preserve boundary context.
- **Local Dense Embeddings:** Leverages HuggingFace's `all-MiniLM-L6-v2` for 384-dimensional vector representations stored in ChromaDB.
- **Offline Inference:** Powered by Ollama (`llama3.2:3b`) with strict grounding prompts to mitigate hallucinations.
- **Interactive Web Interface:** Built with Streamlit for conversational document interaction.

## Tech Stack
- **Frameworks:** LangChain, Streamlit
- **LLM & Embeddings:** Ollama (`llama3.2:3b`), HuggingFace (`all-MiniLM-L6-v2`)
- **Vector Store:** ChromaDB
- **Language:** Python 3.10+
