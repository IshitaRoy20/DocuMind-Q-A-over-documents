# PDF Q&A (RAG) — Streamlit UI, semantic search only

A browser-based version of the PDF Q&A project. Upload PDFs, ask questions,
get answers grounded in the documents with source citations — all through
a chat interface instead of a terminal.

Retrieval is **semantic-only**: embeddings + Chroma vector search. No BM25 /
keyword layer — one retrieval path, simpler to reason about and debug.

## Architecture

```
Browser: upload PDF(s)
   │
   ▼
app.py
   ├─ PyPDFLoader          -> extract text per page
   ├─ RecursiveCharacterTextSplitter -> chunks (1000 chars, 150 overlap)
   ├─ HuggingFaceEmbeddings (all-MiniLM-L6-v2) -> embed each chunk
   └─ Chroma.from_documents -> persisted vector index (chroma_store/)

Chat turn:
   ConversationalRetrievalChain:
     1. condense (chat_history + new question) -> standalone question   [LLM call]
     2. Chroma retriever -> top-k (k=4) most similar chunks             [semantic search]
     3. stuff chunks + standalone question into QA_PROMPT -> answer     [LLM call]
   -> answer + source chunks displayed, chat history updated
```

## Features

- **Drag-and-drop PDF upload** — no manual folder management, multiple files at once
- **Chat interface** — full conversation history stays visible, styled like a normal chat app
- **Conversational memory** — follow-up questions ("what about the second one") resolve correctly
- **Source citations per answer** — expandable "Sources" section shows filename, page number, and the exact snippet the answer was grounded in
- **Sidebar controls** — see which files are indexed, rebuild the index with new files, clear the chat
- **Grounded-only answers** — the prompt forces "I don't have enough information" instead of hallucinating when the PDFs don't contain the answer

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Get a free HuggingFace token: https://huggingface.co/settings/tokens
#    Copy the template and fill in your real token:
cp .env.example .env
#    then edit .env so it contains:
#    HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# 3. Launch the app
streamlit run app.py
```

`app.py` loads `.env` automatically at startup via `python-dotenv`
(`load_dotenv()`), so the token is picked up without needing to `export` it
in your shell every session. `.env` is listed in `.gitignore` so it never
gets committed — only `.env.example` (with a placeholder) should go in
version control.

This opens a browser tab (usually `http://localhost:8501`).

## How to use it

1. In the sidebar, upload one or more PDFs.
2. Click **Build / Rebuild Index** — this chunks + embeds the PDFs (takes a
   few seconds to a minute depending on size).
3. Ask questions in the chat box at the bottom.
4. Expand **Sources** under any answer to see exactly which file/page it
   came from.
5. Upload different PDFs and rebuild the index any time to switch documents.
   Rebuilding also clears the chat, since old chat history wouldn't make
   sense against a different document set.

## Design notes / things worth knowing

- **Why semantic-only**: simpler system, one retrieval path to tune (just
  `TOP_K` and chunk size), and easier to explain end-to-end. The tradeoff
  (documented honestly): exact-term lookups — an error code, an exact name —
  can be missed if no chunk is semantically close enough to the query
  wording. That's the known limitation of dropping BM25.
- **`@st.cache_resource` on the embedding model**: without this, Streamlit
  would reload the ~80MB embedding model on every single interaction
  (Streamlit reruns the whole script on each UI event), which is slow. This
  cache keeps it loaded across reruns within a session.
- **Index rebuilds wipe `chroma_store/`**: this is a deliberate simplicity
  choice for a portfolio project — one active document set at a time, no
  stale-data confusion. A multi-session/multi-user version would need
  per-user or per-upload namespacing instead.
- **Two LLM calls per chat turn** (condense question + generate answer) —
  a real latency cost of conversational memory, not free.

## Natural next steps

- Add a model picker in the sidebar (swap `LLM_REPO_ID` without editing code)
- Persist indexes across sessions instead of wiping on rebuild (e.g. one
  Chroma collection per uploaded file set, selectable from a dropdown)
- Deploy to Streamlit Community Cloud for a shareable public link
