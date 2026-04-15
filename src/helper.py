from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document


#Extract Data From the PDF File
from pathlib import Path

def load_pdf_file(data: str):
    """Walk a directory and load every PDF, skipping any that fail.

    The previous implementation used ``DirectoryLoader`` from LangChain,
    which would attempt to parse every file in a batch.  If pypdf hangs on
    a corrupted/odd PDF the entire process would stall (see the traceback
    in the user's report).  By walking the tree ourselves we can log the
    filename and continue on errors, giving the caller a chance to inspect
    or remove the problematic file.
    """

    documents: list[Document] = []
    root = Path(data)
    if not root.exists():
        raise FileNotFoundError(f"data directory does not exist: {root}")

    for pdf_path in root.rglob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            documents.extend(docs)
        except Exception as exc:
            # print to stderr so it is visible in logs/console
            print(f"[warning] failed to load {pdf_path}: {exc}")
            # optionally continue with next file
            continue

    return documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs



#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks



#Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    return embeddings