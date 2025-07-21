import os
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

import sys
import yaml
import pickle
import fitz  # PyMuPDF
import requests
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import logging
import streamlit as st

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain

# ----------- MUST BE FIRST STREAMLIT CALL --------------
st.set_page_config(page_title="Scheme Research Tool", layout="wide")

# ----------------- Logging ----------------------
if "log_initialized" not in st.session_state:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("scheme_assistant.log", mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("✅ Scheme Assistant Script Started")
    st.session_state.log_initialized = True

# ----------------- Load Config ----------------------
@st.cache_resource
def load_config(path="config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)

config = load_config()
EMBEDDING_MODEL = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
FAISS_INDEX_PATH = config.get("faiss_index_path", "faiss_store.pkl")
LLM_MODEL = config.get("llm_model", "mistral:instruct")

# ----------------- PDF Text Extraction ----------------------
def extract_text_from_pdf_url(url):
    try:
        logging.info(f"📎 Fetching PDF from URL: {url}")
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        pdf_bytes = response.content
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        text = ""
        for page in doc:
            text += page.get_text()

        if not text.strip():
            logging.warning("❗ No text found via PyMuPDF. Switching to OCR.")
            text = extract_text_with_ocr(pdf_bytes)

        logging.info(f"✅ Text extraction completed. Total chars: {len(text)}")
        return text

    except Exception as e:
        logging.error(f"❌ PDF extraction failed: {e}")
        raise ValueError(f"PDF processing failed: {e}")

def extract_text_with_ocr(pdf_bytes):
    logging.info("🔍 Performing OCR on PDF pages")
    images = convert_from_bytes(pdf_bytes)
    text = ""
    for i, img in enumerate(images):
        gray = img.convert("L")
        t = pytesseract.image_to_string(gray)
        logging.info(f"OCR done for page {i+1}, chars: {len(t)}")
        text += t
    return text

# ----------------- FAISS Handling ----------------------
def split_text_into_docs(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    logging.info(f"🧩 Split into {len(chunks)} chunks")
    return [Document(page_content=chunk) for chunk in chunks]

def create_faiss_index(documents, index_path, embedding_model_name):
    logging.info(f"📦 Creating FAISS index using model: {embedding_model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = FAISS.from_documents(documents, embeddings)
    with open(index_path, "wb") as f:
        pickle.dump(vectorstore, f)
    logging.info(f"✅ FAISS index saved to {index_path}")
    return vectorstore

def load_faiss_index(index_path):
    logging.info(f"📂 Loading FAISS index from {index_path}")
    with open(index_path, "rb") as f:
        return pickle.load(f)

# ----------------- Prompt ----------------------
MAIN_PROMPT = """
Provide the following four sections **only**, clearly marked and separated:

1. Scheme Benefits  
2. Application Process  
3. Eligibility  
4. Required Documents

Do not include any introduction or conversational text. Return just the four sections, properly labeled.
"""

# ----------------- Streamlit UI with Sidebar ----------------------

# Sidebar for input and settings
with st.sidebar:
    st.title("🧾 Scheme Assistant")
    st.markdown("Upload a government scheme **PDF URL** and get the summarized:")
    st.markdown("- **Scheme Benefits**\n- **Application Process**\n- **Eligibility**\n- **Required Documents**")
    
    pdf_url = st.text_input("📎 Paste the PDF URL here:")

    st.markdown("---")
    st.markdown("⚙️ **Model Settings**")
    st.write(f"🔢 Embedding: `{EMBEDDING_MODEL}`")
    st.write(f"🧠 LLM: `{LLM_MODEL}`")

st.title("📘 Government Scheme Assistant")
st.markdown("The summarized scheme information will appear below once processed.")

if pdf_url:
    with st.spinner("🔍 Extracting and indexing PDF..."):
        try:
            text = extract_text_from_pdf_url(pdf_url)
            documents = split_text_into_docs(text)
            vectorstore = create_faiss_index(documents, FAISS_INDEX_PATH, EMBEDDING_MODEL)

            st.success("✅ PDF indexed successfully!")
            logging.info("📄 PDF processed and vectorstore ready.")

            llm = Ollama(model=LLM_MODEL)
            chain = load_qa_chain(llm, chain_type="stuff")

            full_prompt = MAIN_PROMPT
            docs = vectorstore.similarity_search(full_prompt)
            answer = chain.run(input_documents=docs, question=full_prompt)

            st.subheader("📑 Scheme Summary")
            st.markdown(answer)
            logging.info(f"🧠 Full summary generated. Length: {len(answer)}")

        except Exception as e:
            logging.error(f"❌ Error in Streamlit flow: {e}")
            st.error(f"Failed: {e}")
