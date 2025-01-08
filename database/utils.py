import os
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from transformers import AutoTokenizer
import pandas as pd
import tqdm

# Function to chunk loaded documents
def chunk_documents(documents: str, chunk_size: int, chunk_overlap: int, tokenizer_name):
    """
    Chunk each document's content into smaller pieces and return a new list of documents.
    
    Args:
        documents (str): The document to be splitted.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between chunks.
    
    Returns:
        list: List of chunked Document objects.
    """

    RAW_KNOWLEDGE_BASE = []
    documents1 = documents.split("{sep}")
    print("Number of documents:", len(documents1))
    len_doc = len(documents1) 
    for ind in range(len_doc):
        if ind%100 == 0:    
            print("Loading document", ind)
        RAW_KNOWLEDGE_BASE.append(LangchainDocument(page_content=documents1[ind]))

    NEWS_SEPARATORS = [
        "\n\n",
        "\n",     # Line breaks
        "\t",     # Tabs
        ". ",     # Sentences
        ", ",     # Clauses within sentences
        "; ",     # Semi-colons separating clauses
        ": ",     # Colons introducing lists or explanations
        " ",      # Spaces
        ""        # No space, as a last resort
        ]
    
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=NEWS_SEPARATORS,
    )

    docs_processed = []
    for j in range(len(RAW_KNOWLEDGE_BASE)):
        if j%100 == 0:
            print(j)
        docs_processed += text_splitter.split_documents([RAW_KNOWLEDGE_BASE[j]])

    return docs_processed

