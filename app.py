import os
import tempfile
import shutil

from dotenv import load_dotenv
import streamlit as st

load_dotenv()  # reads .env in the project root and populates os.environ
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PERSIST_DIR = "chroma_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO_ID = "zai-org/GLM-5.3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

QA_PROMPT = PromptTemplate(
    template="""Use ONLY the following context to answer the question.
If the answer is not contained in the context, say "I don't have enough
information in these documents to answer that." Do not use outside knowledge.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)

st.set_page_config(page_title="PDF Q&A (RAG)", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources — avoid reloading the embedding model on every rerun
# ---------------------------------------------------------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_llm():
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        st.error(
            "HUGGINGFACEHUB_API_TOKEN is not set. "
            "Add it to a .env file in the project root — see README."
        )
        st.stop()
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.2,
        huggingfacehub_api_token=token,
    )
    return ChatHuggingFace(llm=llm_endpoint)


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------
def build_index(uploaded_files) -> tuple[Chroma, list[str]]:
    """Save uploaded PDFs to a temp dir, chunk + embed them, return the store."""
    embeddings = get_embeddings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    # wipe any previous index so re-uploading gives a clean slate
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    all_chunks = []
    filenames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for uploaded_file in uploaded_files:
            path = os.path.join(tmpdir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(path)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = uploaded_file.name
                d.metadata["page"] = d.metadata.get("page", 0) + 1  # 1-indexed

            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
            filenames.append(uploaded_file.name)

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    return vectorstore, filenames


def build_chain(vectorstore: Chroma):
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )
    return chain


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "chain" not in st.session_state:
    st.session_state.chain = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "sources"}


# ---------------------------------------------------------------------------
# Sidebar — upload & index controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type="pdf", accept_multiple_files=True
    )

    if st.button("Build / Rebuild Index", type="primary", disabled=not uploaded_files):
        with st.spinner("Chunking and embedding documents..."):
            vectorstore, filenames = build_index(uploaded_files)
            st.session_state.chain = build_chain(vectorstore)
            st.session_state.indexed_files = filenames
            st.session_state.messages = []  # fresh chat for a fresh index
        st.success(f"Indexed {len(filenames)} file(s).")

    if st.session_state.indexed_files:
        st.subheader("Indexed files")
        for f in st.session_state.indexed_files:
            st.caption(f"• {f}")

    st.divider()

    if st.button("Clear chat", disabled=st.session_state.chain is None):
        st.session_state.messages = []
        if st.session_state.chain is not None:
            st.session_state.chain.memory.clear()
        st.rerun()

    st.divider()
    st.caption(
        "Retrieval: semantic search only (embeddings via "
        f"`{EMBEDDING_MODEL}`, stored in Chroma). Answers are grounded "
        "strictly in the uploaded PDFs."
    )


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("PDF Q&A — RAG Chat")

if st.session_state.chain is None:
    st.info("👈 Upload one or more PDFs and click **Build / Rebuild Index** to start.")
else:
    # replay chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources"):
                    for doc in msg["sources"]:
                        src = doc.metadata.get("source", "?")
                        page = doc.metadata.get("page", "?")
                        snippet = doc.page_content[:200].replace("\n", " ")
                        st.markdown(f"**{src}** — page {page}")
                        st.caption(f"\"{snippet}...\"")

    # new question
    question = st.chat_input("Ask a question about your documents...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chain.invoke({"question": question})
                answer = result["answer"]
                sources = result["source_documents"]
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for doc in sources:
                        src = doc.metadata.get("source", "?")
                        page = doc.metadata.get("page", "?")
                        snippet = doc.page_content[:200].replace("\n", " ")
                        st.markdown(f"**{src}** — page {page}")
                        st.caption(f"\"{snippet}...\"")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
