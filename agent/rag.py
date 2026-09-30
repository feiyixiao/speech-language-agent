import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

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

    # Build from documents
    headers_to_split = [("#", "section")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split)

    docs = []
    for fname in os.listdir(DOCS_PATH):
        if fname.endswith(".md"):
            fpath = os.path.join(DOCS_PATH, fname)
            with open(fpath) as f:
                text = f.read()
            splits = splitter.split_text(text)
            for split in splits:
                split.metadata["source"] = fname
            docs.extend(splits)

    _vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_PATH)
    return _vectorstore


rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a language teacher. Answer the student's question using ONLY the context below. "
               "If the context does not contain the answer, say 'I don't have that in my knowledge base, but here's what I know: ' "
               "and give a brief general answer.\n\nContext:\n{context}"),
    ("human", "{question}")
])

def answer_grammar_question(question: str) -> str:
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join(d.page_content for d in docs)
    llm = ChatGroq(model="llama-3.1-8b-instant")
    chain = rag_prompt | llm
    result = chain.invoke({"context": context, "question": question})
    return result.content
