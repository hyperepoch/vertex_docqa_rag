### Vertex AI Document Q&A (RAG) Scaffold
# ───────────────────────────────────────
# 📁 Structure
# - main.py               ← Entry point for Streamlit app
# - utils/embedding.py    ← Embedding functions using Vertex AI
# - utils/vector_store.py ← FAISS-based vector store logic
# - utils/parser.py       ← Document parsing and chunking
# - utils/storage.py      ← GCS integration helpers
# - tests/                ← Unit tests for core logic
# - requirements.txt      ← Required packages
# - Dockerfile            ← For Cloud Run deployment
# - app.yaml              ← For App Engine deployment
# - cloudbuild.yaml       ← CI/CD automation for Cloud Build

# ───────────────────────────────────────
# 📁 main.py
import streamlit as st
import os
from utils.embedding import get_embedding, query_with_context
from utils.parser import parse_and_chunk
from utils.vector_store import VectorStore
from utils.storage import upload_to_gcs, download_from_gcs

# 🔐 Optional password protection
if os.getenv("STREAMLIT_PASSWORD"):
    pw = st.text_input("🔐 Enter password", type="password")
    if pw != os.getenv("STREAMLIT_PASSWORD"):
        st.stop()

st.title("📄 Document Q&A with Vertex AI")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "docx"])
question = st.text_input("Ask a question about the document:")

download_option = st.checkbox("⬇️ Download processed file from GCS")
if download_option:
    gcs_file = st.text_input("Enter GCS filename (e.g., mydoc.pdf):")
    if gcs_file and st.button("Download"):
        st.write(download_from_gcs(gcs_file))

if uploaded_file:
    file_bytes = uploaded_file.read()
    st.success("Document uploaded. Parsing and embedding...")
    chunks = parse_and_chunk(file_bytes, uploaded_file.type)
    embeddings = [get_embedding(chunk) for chunk in chunks]

    upload_to_gcs(uploaded_file.name, file_bytes)

    if "vs" not in st.session_state:
        st.session_state.vs = VectorStore()
        st.session_state.vs.add(chunks, embeddings)

    if question:
        top_k_chunks = st.session_state.vs.query(question, k=5)
        answer = query_with_context(question, top_k_chunks)
        st.write("### 📌 Answer")
        st.success(answer)

# ───────────────────────────────────────
# 📁 utils/embedding.py
from vertexai.preview.language_models import TextEmbeddingModel, TextGenerationModel

embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
generation_model = TextGenerationModel.from_pretrained("text-bison@002")

def get_embedding(text):
    return embedding_model.get_embeddings([text])[0].values

def query_with_context(question, top_chunks):
    context = "\n".join(top_chunks)
    prompt = f"""Use the following context to answer the question:
{context}

Question: {question}
Answer:"""
    response = generation_model.predict(prompt=prompt, temperature=0.2, max_output_tokens=500)
    return response.text

# ───────────────────────────────────────
# 📁 utils/vector_store.py
import faiss
import numpy as np
from utils.embedding import get_embedding

class VectorStore:
    def __init__(self):
        self.embeddings = []
        self.chunks = []
        self.index = None

    def add(self, chunks, embeddings):
        self.chunks = chunks
        self.embeddings = np.array(embeddings).astype("float32")
        self.index = faiss.IndexFlatL2(len(embeddings[0]))
        self.index.add(self.embeddings)

    def query(self, question, k=5):
        q_emb = np.array([get_embedding(question)]).astype("float32")
        _, I = self.index.search(q_emb, k)
        return [self.chunks[i] for i in I[0]]

# ───────────────────────────────────────
# 📁 utils/parser.py
import fitz  # PyMuPDF
import docx
from google.cloud import documentai_v1beta3 as documentai
import io

PROJECT_ID = "your-project-id"  # Replace with your project ID
LOCATION = "us"
PROCESSOR_ID = "your-processor-id"  # Replace with your processor ID

def parse_and_chunk(file_bytes, file_type, chunk_size=500):
    text = ""
    if file_type == "application/pdf":
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text()
        if not text.strip():
            text = run_document_ai(file_bytes)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        text = file_bytes.decode("utf-8")

    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def run_document_ai(file_bytes):
    client = documentai.DocumentUnderstandingServiceClient()
    name = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"

    raw_document = documentai.RawDocument(content=file_bytes, mime_type="application/pdf")
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    result = client.process_document(request=request)
    return result.document.text

# ───────────────────────────────────────
# 📁 utils/storage.py
from google.cloud import storage
import tempfile

def upload_to_gcs(filename, file_bytes, bucket_name="your-bucket-name"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(filename)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    blob.upload_from_filename(tmp_path)

def download_from_gcs(file_name, bucket_name="your-bucket-name"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    return blob.download_as_text()

# ───────────────────────────────────────
# 📁 tests/test_parser.py
import io
from utils import parser

def test_parse_txt():
    file = io.BytesIO(b"This is a test text document for parsing.")
    file_type = "text/plain"
    chunks = parser.parse_and_chunk(file.read(), file_type, chunk_size=5)
    assert chunks == ["This is a test text", "document for parsing."]

def test_parse_docx():
    # Skipping actual docx test as it needs a real file instance
    pass

# ───────────────────────────────────────
# 📁 tests/test_vector_store.py
from utils.vector_store import VectorStore
import numpy as np

def test_vector_store_add_and_query():
    vs = VectorStore()
    chunks = ["test chunk one", "test chunk two"]
    embeddings = [np.ones(768), np.ones(768) * 2]
    vs.add(chunks, embeddings)
    result = vs.query("test", k=1)
    assert isinstance(result, list) and len(result) == 1

# ───────────────────────────────────────
# 📁 requirements.txt
streamlit
PyMuPDF
faiss-cpu
google-cloud-aiplatform
google-cloud-storage
python-docx
google-cloud-documentai

# ───────────────────────────────────────
# 📁 Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.enableCORS=false"]

# ───────────────────────────────────────
# 📁 app.yaml
runtime: python310
entrypoint: streamlit run main.py --server.port=$PORT

# ───────────────────────────────────────
# 📁 cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/vertex-doc-qa', '.']
  - name: 'gcr.io/cloud-builders/gcloud'
    args: ['run', 'deploy', 'vertex-doc-qa', '--image', 'gcr.io/$PROJECT_ID/vertex-doc-qa', '--region', 'us-central1', '--platform=managed', '--allow-unauthenticated']

images:
  - 'gcr.io/$PROJECT_ID/vertex-doc-qa'

# ───────────────────────────────────────
# ✅ Deployment Walkthrough:
# 1. Replace project settings in `parser.py` and `storage.py`
# 2. Set a password (optional): `export STREAMLIT_PASSWORD="your_pw"`
# 3. Enable APIs: Vertex AI, Document AI, Cloud Run, Cloud Build, GCS
# 4. Run locally: `streamlit run main.py`
# 5. Deploy with Cloud Build:
#    gcloud builds submit --config cloudbuild.yaml
# 6. Access your app via the Cloud Run or App Engine URL