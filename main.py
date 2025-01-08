import RAG.RAG as rag
import google.generativeai as genai
from huggingface_hub import login
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from flask import Flask, request, render_template, Blueprint
import torch

model = genai.GenerativeModel("models/gemini-1.5-flash-002")
faiss_folder = os.get_env("FAISS_FOLDER")

# if you use another embedding model, modify here.
embedding_model = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            multi_process=True,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
            show_progress=True,)

knowledge_index = FAISS.load_local(
        faiss_folder, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
app = Flask(__name__, template_folder = 'template')

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/getprediction',methods=['POST'])
def getprediction():    

   input = request.form['question']
   #prediction = rag.answer_without_context(input, model)
   prediction = rag.answer_one_sample(input, model, knowledge_index)

   return render_template('index.html', output=prediction)
   

if __name__ == "__main__":
    app.run(debug=True)

