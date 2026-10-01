# 🏦 Hybrid RAG-Based Banking Research Agent

A production-style **AI-powered banking research assistant** that uses **Hybrid Retrieval-Augmented Generation (RAG)** to answer questions from bank annual reports and financial documents.

The system combines **dense vector search using Qdrant** with **sparse BM25 retrieval**, merges results using **Reciprocal Rank Fusion (RRF)**, and improves relevance using a **Cross-Encoder Reranker** before generating the final answer with a **Groq LLM**.

It provides source-aware answers with document and page references through a **FastAPI backend** and an interactive **Streamlit frontend**.

---

## 🚀 Project Overview

Financial annual reports contain large amounts of structured and unstructured information such as:

* Revenue
* Net profit
* Deposits
* Advances
* NPA
* Capital Adequacy Ratio
* ROA
* ROE
* Financial performance
* Risk management
* Business growth
* Management discussion
* Balance sheet information

Finding specific information manually from hundreds of pages can be time-consuming.

This project solves that problem by allowing users to ask questions in natural language.

### Example

```text
User:
What was HDFC Bank's net profit in FY2025?

AI:
HDFC Bank reported a net profit of ...

Sources:
- HDFC_Bank_Annual_Report_2025.pdf
- Page 42
```

The system retrieves relevant information from the uploaded annual reports before generating the answer.

---

# 🧠 Architecture

```text
                    ┌──────────────────────┐
                    │    Bank Reports      │
                    │      PDF Files       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   PDF Document Load  │
                    │       PyPDF           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Document Chunking   │
                    │      LangChain       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Embedding Generation │
                    │ Sentence Transformers│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │        Qdrant        │
                    │   Vector Database    │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    │                      │
                    ▼                      ▼
             Dense Retrieval        Sparse Retrieval
                    │                   BM25
                    │                      │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Hybrid Retrieval     │
                    │        RRF           │
                    │ Reciprocal Rank      │
                    │      Fusion          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Cross-Encoder        │
                    │ Reranker             │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Relevant Context     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Groq LLM        │
                    │ openai/gpt-oss-120b  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Answer + Citations   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      FastAPI         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Streamlit        │
                    │     Frontend         │
                    └──────────────────────┘
```

---

# 🔥 Key Features

## 1. 📄 PDF Annual Report Processing

The system can process bank annual reports in PDF format.

It automatically:

* Loads PDF documents
* Extracts text
* Splits documents into smaller chunks
* Preserves document metadata
* Stores source filename
* Stores page information

---

## 2. 🔢 Semantic Embeddings

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

to convert document chunks and user queries into numerical vector representations.

These embeddings allow the system to find documents based on **meaning**, not only exact keywords.

---

## 3. 🗄️ Qdrant Vector Database

**Qdrant** is used as the primary vector database.

It stores:

```text
Vector Embeddings
        +
Document Text
        +
Metadata
        +
Source Information
        +
Page Number
```

The project uses vector similarity search to retrieve semantically relevant document chunks.

### Why Qdrant?

* High-performance vector search
* Efficient similarity search
* Metadata filtering
* Persistent vector storage
* Easy local Docker deployment
* Suitable for RAG applications
* Can scale independently from the application

---

# 🔎 4. Hybrid Retrieval

The project does not rely only on vector search.

It combines two retrieval strategies:

### Dense Retrieval

Uses:

```text
User Query
     ↓
Embedding Model
     ↓
Qdrant
     ↓
Semantic Similarity Search
```

This helps retrieve conceptually similar information.

### Sparse Retrieval

Uses:

```text
User Query
     ↓
BM25
     ↓
Keyword-Based Retrieval
```

BM25 is particularly useful for financial terminology such as:

```text
NPA
ROA
ROE
CAR
Net Profit
Deposits
Advances
Capital Adequacy
```

---

# 🔀 5. Reciprocal Rank Fusion (RRF)

The results from Qdrant and BM25 are combined using **Reciprocal Rank Fusion**.

Conceptually:

```text
Qdrant Results
      +
BM25 Results
      ↓
     RRF
      ↓
Combined Ranking
```

RRF helps combine semantic and keyword-based retrieval without relying on only one retrieval method.

Formula:

```text
RRF(d) = Σ 1 / (k + rank(d))
```

where:

* `d` = document
* `k` = ranking constant
* `rank(d)` = document ranking position

---

# 🎯 6. Cross-Encoder Reranking

After hybrid retrieval, the retrieved candidates are passed through a Cross-Encoder.

Model:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The model evaluates:

```text
(Query, Document Chunk)
```

and produces a relevance score.

The highest-relevance chunks are selected as context for the LLM.

---

# 🤖 7. Groq LLM

The project uses the Groq API for fast LLM inference.

Default model:

```text
openai/gpt-oss-120b
```

The LLM receives the retrieved context and generates an answer based on the available documents.

The system is designed to reduce hallucination by grounding responses in retrieved document content.

---

# 📚 8. Source Citations

The system preserves document metadata during ingestion.

Example:

```text
Source:
HDFC_Bank_Annual_Report_2025.pdf

Page:
42
```

This allows generated answers to provide references back to the original annual report.

---

# ⚡ 9. FastAPI Backend

FastAPI provides the backend API layer.

Example endpoint:

```text
POST /ask
```

Example request:

```json
{
  "query": "What was the net profit of HDFC Bank in FY2025?"
}
```

Example:

```text
User Query
    ↓
FastAPI
    ↓
Hybrid RAG Pipeline
    ↓
Groq
    ↓
Answer
```

Swagger API documentation is available at:

```text
http://localhost:8000/docs
```

---

# 🖥️ 10. Streamlit Interface

Streamlit provides the user-facing interface.

Users can:

* Enter financial questions
* Query annual reports
* View generated answers
* View retrieved sources
* Interact with the RAG system without using the API directly

---

# 📁 Project Structure

```text
Hybrid_RAG_Based_Banking_Research_Agent/
│
├── backend/
│   │
│   │
│   ├── api.py
│   │   └── FastAPI application and API endpoints
│   │
│   ├── chat_model.py
│   │   └── Groq LLM configuration
│   │
│   ├── embedding.py
│   │   └── Sentence Transformer embedding generation
│   │
│   ├── fusion.py
│   │   └── Reciprocal Rank Fusion implementation
│   │
│   ├── generate.py
│   │   └── Answer generation and citation handling
│   │
│   ├── ingest.py
│   │   └── PDF ingestion and indexing pipeline
│   │
│   ├── pipeline.py
│   │   └── Complete Hybrid RAG pipeline
│   │
│   ├── reranker.py
│   │   └── Cross-Encoder reranking
│   │
│   ├── sparse_retriever.py
│   │   └── BM25 sparse retrieval
│   │
│   └── vector_store.py
│       └── Qdrant vector database operations
│
├── data/
│      └── Place annual report PDFs here
│
├── frontend.py
│   └── Streamlit frontend
│
├── requirements.txt
│   └── Python dependencies
│
├── Dockerfile
│   └── Application Docker configuration
│
├── docker-compose.yml
│   └── Qdrant + API + Streamlit services
│
├── .env.example
│   └── Environment variable template
│
├── .gitignore
│
└── README.md
```

---

# 🛠️ Technologies Used

| Technology                | Purpose                                |
| ------------------------- | -------------------------------------- |
| **Python**                | Core programming language              |
| **LangChain**             | Document processing and text splitting |
| **Qdrant**                | Vector database and semantic search    |
| **Sentence Transformers** | Text embeddings                        |
| **BM25**                  | Sparse keyword retrieval               |
| **RRF**                   | Hybrid retrieval result fusion         |
| **Cross-Encoder**         | Document reranking                     |
| **Groq**                  | LLM inference                          |
| **Llama 3.3 70B**         | Generative AI model                    |
| **FastAPI**               | Backend REST API                       |
| **Streamlit**             | Frontend/UI                            |
| **PyPDF**                 | PDF text extraction                    |
| **Docker**                | Containerization                       |
| **Docker Compose**        | Multi-service deployment               |
| **python-dotenv**         | Environment configuration              |

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/priyanshukushwaha0/Hybrid RAG-Based Banking Research Agent.git
```

```bash
cd Hybrid RAG-Based Banking Research Agent
```

---

# 🔐 2. Configure Environment Variables

Create a `.env` file:

```bash
cp .env.example .env
```

On Windows:

```powershell
copy .env.example .env
```

Add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here

GROQ_MODEL= openai/gpt-oss-120b

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=bank_documents

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

BM25_INDEX_PATH=data/bm25_index.pkl

API_HOST=0.0.0.0
API_PORT=8000
```

---

# 📄 3. Add Bank Annual Reports

Place your PDF files inside:

```text
data/
```

Example:

```text
data/
├── HDFC_Bank_Annual_Report_2025.pdf
├── ICICI_Bank_Annual_Report_2025.pdf
├── SBI_Annual_Report_2025.pdf
└── Axis_Bank_Annual_Report_2025.pdf
```

---

# 🐳 4. Start Qdrant

The easiest way to run Qdrant locally is Docker.

```bash
docker compose up -d qdrant
```

Qdrant will be available at:

```text
http://localhost:6333
```

Qdrant dashboard:

```text
http://localhost:6333/dashboard
```

---

# 🐍 5. Create Virtual Environment

### Windows

```bash
python -m venv myenv
```

```bash
.venv\Scripts\activate

---

# 📦 6. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📥 7. Ingest Documents

Run:

```bash
python -m backend.ingest
```

The ingestion pipeline performs:

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embedding Generation
 ↓
Qdrant
 ↓
BM25 Index
```

After ingestion, your documents are ready for retrieval.

---

# 🚀 8. Start FastAPI

Run:

```bash
uvicorn backend.api:app --reload
```

API:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

# 🖥️ 9. Start Streamlit

Open another terminal:

```bash
streamlit run frontend.py
```

Open:

```text
http://localhost:8501
```

---

# 🐳 Run the Complete Application with Docker

You can run the complete application using:

```bash
docker compose up --build
```

Services:

```text
Qdrant       → http://localhost:6333
FastAPI      → http://localhost:8000
Swagger      → http://localhost:8000/docs
Streamlit    → http://localhost:8501
```

---

# 🔄 RAG Pipeline

The complete question-answering process is:

```text
                User Question
                      │
                      ▼
               Streamlit UI
                      │
                      ▼
                 FastAPI
                      │
                      ▼
             Query Processing
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
    Dense Retrieval         Sparse Retrieval
       Qdrant                    BM25
          │                       │
          └───────────┬───────────┘
                      ▼
                    RRF
                      │
                      ▼
              Hybrid Results
                      │
                      ▼
             Cross-Encoder
                Reranker
                      │
                      ▼
             Top Relevant Chunks
                      │
                      ▼
                 Groq LLM
                      │
                      ▼
              Final Answer
                      │
                      ▼
            Source Citations
```

---

# 💡 Example Questions

After uploading annual reports, you can ask:

```text
What was HDFC Bank's net profit in FY2025?
```

```text
What was the bank's capital adequacy ratio?
```

```text
How did deposits change during FY2025?
```

```text
What was the growth in advances?
```

```text
What was the bank's ROA?
```

```text
What were the major risk factors mentioned in the annual report?
```

```text
Compare the financial performance of two banks based on their annual reports.
```

---

# 🔐 Environment Variables

| Variable            | Description                |
| ------------------- | -------------------------- |
| `GROQ_API_KEY`      | Groq API key               |
| `GROQ_MODEL`        | Groq LLM model             |
| `QDRANT_URL`        | Qdrant server URL          |
| `QDRANT_COLLECTION` | Qdrant collection name     |
| `EMBEDDING_MODEL`   | Sentence Transformer model |
| `RERANKER_MODEL`    | Cross-Encoder model        |
| `BM25_INDEX_PATH`   | BM25 index location        |
| `API_HOST`          | FastAPI host               |
| `API_PORT`          | FastAPI port               |

---
