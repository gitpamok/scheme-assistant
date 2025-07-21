# scheme-assistant
# 📘 Scheme Research Tool

This is a Streamlit-based application (`scheme_assistant.py`) that extracts and summarizes key sections from **government scheme PDFs** via URL. It leverages **OCR fallback**, **FAISS vector similarity**, **HuggingFace embeddings**, and **LLMs (via Ollama)** to output:

- ✅ Scheme Benefits  
- 📝 Application Process  
- 🎯 Eligibility  
- 📄 Required Documents

---

## 🚀 Features

- 🧾 PDF text extraction (with OCR fallback using Tesseract)
- 🧠 Summarization using an Ollama-hosted LLM (e.g., `mistral:instruct`)
- 🔍 Semantic chunking with FAISS vector search
- ⚙️ Easily configurable via `config.yaml`
- 💡 Fully interactive UI via Streamlit

---

## 🧰 Requirements

- Python 3.8+
- `ollama` running locally (`http://127.0.0.1:11434`)
- `tesseract-ocr` installed and accessible via system PATH
- `poppler` installed (required by `pdf2image`)

---

### Installation and Running

```bash

# Install Python dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run scheme_assistant.py

