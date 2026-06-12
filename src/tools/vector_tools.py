from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DEFAULT_PERSIST_DIR = "vectorstore"
DEFAULT_COLLECTION_NAME = "legal_corpus"

def build_index(
        documents: list[dict[str, Any]],
        persist_directory: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
) -> Chroma:
    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )

    langchain_docs: list[Document] = []

    for doc in documents:
        chunks = splitter.split_text(doc["text"])

        for idx, chunks in enumerate(chunks):
            langchain_docs.append(
                Document(
                    page_content=chunks,
                    metadata={**doc["metadata"], "chunk_id": idx,
                    },
                )
            )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    return Chroma.from_documents(
        documents=langchain_docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

def load_vectorstore(
        persist_directory: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    return Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embeddings,
    )