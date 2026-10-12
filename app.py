import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Personal Local RAG", layout="wide")
st.title("📄 100% Local RAG Assistant for Discussion(Ollama)")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Local Model Settings")
    model_name = st.selectbox(
        "Choose Ollama Model:",
        ["llama3.2:3b", "llama3.2", "llama3", "mistral", "phi3"]
    )
    st.caption("Make sure this model is pulled in your terminal (e.g. `ollama pull llama3.2`).")

    st.markdown("---")
    st.subheader("Data Ingestion")
    source_type = st.radio("Source:", ["Upload PDF", "Website URL"])
    
    docs = []
    if source_type == "Upload PDF":
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_file and st.button("Process Document"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                loader = PyPDFLoader(tmp.name)
                docs = loader.load()
    else:
        url_input = st.text_input("Paste Webpage URL:")
        if url_input and st.button("Fetch & Process"):
            loader = WebBaseLoader(url_input)
            docs = loader.load()

    # Process into Vector Database
    if docs:
        with st.spinner("Chunking & embedding text locally..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            splits = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=embeddings,
                collection_name="local_rag_collection"
            )
            st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            st.success("Document ready! You can now ask questions.")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_query = st.chat_input("Ask a question about the document...")

if user_query:
    if "retriever" not in st.session_state:
        st.warning("Please upload a PDF or enter a website link first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            # Local Ollama LLM - No API Key needed!
            llm = ChatOllama(
                model=model_name,
                temperature=0.2
            )

            prompt = ChatPromptTemplate.from_template("""
            Answer the user's question strictly based on the following context.
            If the answer cannot be found in the context, say "I dotn have enough info to ans this."
            
            <context>
            {context}
            </context>
            
            Question: {input}
            """)

            doc_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(st.session_state.retriever, doc_chain)

            with st.spinner("Searching and thinking locally..."):
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})