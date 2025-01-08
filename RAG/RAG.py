import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Pipeline
from typing import List, Tuple
import torch
import json
import tqdm
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document as LangchainDocument
from RAG.utils import load_reader_model, create_prompt_template, gemini_prompt
import google.generativeai as genai

def answer_without_context(question: str, model) -> str:
    """
    Function to answer a question without context
    """
    response = model.generate_content(
                contents = question,
                generation_config=genai.GenerationConfig(
                temperature = 1.0, max_output_tokens = 4096))

    return response.text


def answer_one_sample(question: str, model, knowledge_index: FAISS, 
                      num_retrieved_docs: int = 10,) -> str:
    """
    Function to answer a question with RAG
    """
    
    # Retrieve documents
    relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
    relevant_docs_content = [doc.page_content for doc in relevant_docs]

    # Build the context from retrieved documents
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc + "\n" for i, doc in enumerate(relevant_docs_content)])


    response = model.generate_content(
                contents = gemini_prompt(question, context),
                generation_config=genai.GenerationConfig(
                response_mime_type="application/json", temperature = 1.0, max_output_tokens = 4096))


    return response.text
