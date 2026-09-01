# DocuMind — RAG-Powered PDF Q&A

Ask questions about your PDFs and get answers grounded in the actual content — not hallucinated, not generic. Upload documents, chat naturally, and see exactly which page every answer came from.

Built with LangChain, HuggingFace, and Chroma, wrapped in a Streamlit chat interface.

---

## What it does

1. **Upload** one or more PDFs through the browser
2. **Ask questions** in a normal chat interface
3. **Get grounded answers** — every response is generated only from retrieved content, with a citation back to the exact file and page it came from
4. **Follow up naturally** — conversational memory means "what about the second one?" resolves correctly across turns

If the answer isn't in your documents, DocuMind says so instead of making something up.

---

## Features

| Feature | Description |
|---|---|
| 📄 Multi-PDF upload | Index any number of PDFs at once, directly from the browser |
| 💬 Conversational chat | Full chat history + memory — follow-up questions work naturally |
| 🔍 Semantic search | Embedding-based retrieval finds conceptually relevant content, not just keyword matches |
| 📌 Source citations | Every answer shows the exact file, page, and text snippet it's grounded in |
| 🚫 No hallucination by design | Prompt strictly constrains answers to retrieved context |
| 🔄 Rebuildable index | Swap documents anytime — rebuild the index with new files in seconds |

---

## How it works

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌─────────────┐
│  Upload PDF  │ ──> │  Chunk text   │ ──> │  Embed chunks  │ ──> │  Chroma DB   │
└─────────────┘     └──────────────┘     └───────────────┘     └─────────────┘
                                                                        │
┌─────────────┐     ┌──────────────┐     ┌───────────────┐            │
│   Answer +   │ <── │  LLM generates│ <── │ Retrieve top-k │ <──────────┘
│   sources    │     │   response    │     │  similar chunks│
└─────────────┘     └──────────────┘     └───────────────┘
        ▲                                          ▲
        │              ┌──────────────┐            │
        └───────────── │ Chat history  │ ───────────┘
                        │ (memory)      │  (standalone question
                        └──────────────┘   condensed from history)
```

**Pipeline in detail:**
1. `PyPDFLoader` extracts text per page from each uploaded PDF
2. `RecursiveCharacterTextSplitter` breaks text into overlapping chunks (1000 chars, 150 overlap)
3. `sentence-transformers/all-MiniLM-L6-v2` (via HuggingFace) embeds each chunk
4. Chroma stores the embeddings as a searchable local vector index
5. On each question: chat history is condensed into a standalone question, top-4 similar chunks are retrieved, and an LLM generates an answer using only that retrieved context

---

## Tech stack

- **UI**: Streamlit
- **Orchestration**: LangChain
- **Embeddings**: HuggingFace `sentence-transformers`
- **LLM inference**: HuggingFace Inference API
- **Vector store**: Chroma (local, persisted to disk)
- **PDF parsing**: `pypdf`

---

## Setup

### 1. Clone / download the project
```bash
cd documind
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get a free HuggingFace API token
Create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — make sure **"Make calls to Inference Providers"** permission is enabled.

### 5. Configure your `.env` file
```bash
cp .env.example .env
```
Edit `.env`:
```dotenv
HUGGINGFACEHUB_API_TOKEN="hf_your_token_here"
```

### 6. Run the app
```bash
streamlit run app.py
```
Opens automatically at `http://localhost:8501`.

---

## Usage

1. Upload one or more PDFs in the sidebar
2. Click **Build / Rebuild Index**
3. Ask a question in the chat box
4. Expand **Sources** under any answer to see the citation
5. Use **Clear chat** to reset the conversation, or upload new PDFs and rebuild to switch documents entirely

---

## Design decisions & limitations

- **Semantic search only** — no keyword/BM25 layer, by design, for a simpler single-path retrieval system. Trade-off: exact-term lookups (an error code, an exact name) may be missed if no chunk is semantically close to the query wording.
- **Two LLM calls per turn** — one to condense chat history into a standalone question, one to generate the answer. This is what powers follow-up questions, but it roughly doubles latency/cost versus a single-turn Q&A system.
- **Index rebuilds wipe the previous one** — one active document set at a time, by design, to avoid stale-data confusion in a single-user local tool.
- **Default LLM**: `HuggingFaceH4/zephyr-7b-beta` (ungated, works out of the box). Swap `LLM_REPO_ID` in `app.py` for a different HuggingFace-hosted instruct model if desired — note gated models (e.g. Meta Llama) require requesting access first.

---

## Roadmap / possible extensions

- [ ] Cross-encoder reranking after retrieval for higher answer precision
- [ ] Support for additional file types (docx, txt, web pages)
- [ ] Persist multiple document sets, selectable from a dropdown
- [ ] Deploy to Streamlit Community Cloud for a shareable link

---

## Project structure
```
documind/
├── app.py              # Full application: ingestion, RAG chain, and UI
├── requirements.txt     # Pinned dependencies
├── .env.example          # Template for required environment variables
├── .gitignore
└── README.md
```
