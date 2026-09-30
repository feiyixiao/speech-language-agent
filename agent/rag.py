"""Grammar knowledge base: markdown docs split by '#' header -> Chroma."""
import os

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

DOCS_PATH = os.path.join(os.path.dirname(__file__), "../data/grammar_docs")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "../data/chroma_db")

_vectorstore = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        _vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        return _vectorstore

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "section")])
    docs = []
    for fname in sorted(os.listdir(DOCS_PATH)):
        if fname.endswith(".md"):
            with open(os.path.join(DOCS_PATH, fname)) as f:
                splits = splitter.split_text(f.read())
            for split in splits:
                split.metadata["source"] = fname
            docs.extend(splits)

    _vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_PATH)
    return _vectorstore


def retrieve(question: str, k: int = 3):
    return get_vectorstore().similarity_search(question, k=k)
