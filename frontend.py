import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Banking Research Agent",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Hybrid RAG-Based Banking Research Agent")
st.caption("Normal Chat + Banking Annual-Report Hybrid RAG")

with st.sidebar:
    st.header("Agent Mode")

    mode_label = st.radio(
        "Choose mode",
        ["Auto", "Banking Research", "Normal Chat"],
    )

    mode = {
        "Auto": "auto",
        "Banking Research": "banking",
        "Normal Chat": "chat",
    }[mode_label]

    top_k = st.slider("Retrieved report chunks", 1, 10, 5)

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        """
### Architecture
PDF Annual Reports → LangChain → Embeddings → Qdrant

BM25 + Vector Search → RRF → Cross-Encoder → Groq → Citations

### Modes
- **Auto:** banking questions use RAG; other questions use normal chat.
- **Banking Research:** always uses annual-report RAG.
- **Normal Chat:** normal Groq conversation.
"""
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        sources = message.get("sources", [])
        if sources:
            with st.expander("📚 Sources"):
                for source in sources:
                    st.write(
                        f"📄 {source['source']} — page {source['page']}"
                    )

query = st.chat_input("Ask anything...")

if query:
    st.session_state.messages.append({
        "role": "user",
        "content": query,
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "query": query,
                        "mode": mode,
                        "top_k": top_k,
                    },
                    timeout=300,
                )

                try:
                    data = response.json()
                except ValueError:
                    data = {"error": response.text}

                if not response.ok:
                    error = data.get(
                        "error",
                        data.get("detail", response.text),
                    )
                    error_type = data.get(
                        "error_type",
                        "BackendError",
                    )
                    st.error(f"{error_type}: {error}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Backend error: {error}",
                    })
                else:
                    answer = data.get(
                        "answer",
                        "No answer returned.",
                    )
                    sources = data.get("sources", [])

                    st.markdown(answer)

                    if data.get("mode") == "banking_rag":
                        st.caption("🔎 Banking Hybrid RAG")
                    else:
                        st.caption("💬 Normal Chat")

                    if sources:
                        with st.expander("📚 Retrieved Sources"):
                            for source in sources:
                                st.write(
                                    f"📄 {source['source']} — page {source['page']}"
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

            except requests.RequestException as exc:
                error = f"Cannot connect to FastAPI: {exc}"
                st.error(error)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error,
                })
