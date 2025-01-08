# **RAG Application for Legal Document**

In this project, i use crawled documents from Thuvienphapluat to form a knowledge database in FAISS and build a retrieval-augmented generation application for these documents.

The documents are transformed to a vector database using RecursiveCharacterTextSplitter (the embedding model, chunk size, chunk overlap can be edited). I utilize FAISS for the vector database and Langchain for chunking. For the generator, i use Gemini models using API (in this code i use 1.5 Flash, which can be modified in main.py). The app is built using Flask as backend and a simple (i'm not humble, it's simple af) HTML frontend.

# How to run
You can run database\create_knowledge_database.ipynb to obtain the FAISS folder of the knowledge database. The embedding model, chunk size and chunk overlap size can be edited there.

The crawled documents and a database folder (using multilingual-e5-base as embedding model, a chunk size of 512 and chunk overlap size of 100) i ran can be found in [this drive link](https://drive.google.com/drive/folders/1r7hWy5v0baKppk2sMGAWLS0yiEoD0z3-?usp=sharing). The FAISS database folder, Huggingface token and Gemini API key should be defined using OS variables.

**Google Colab**

Since my personal device does not have a powerful GPU, i run the app using Google Colab. You can upload the notebook main.ipynb on that environment and run the app. 

**Local**
If you're running on your PC, the app can be run using a virtual environment:

```python
python -m venv venv
venv\Scripts\activate
```

Install the required libraries:
```python
pip install -r requirements.txt
```

Remember to set necessary OS variables.

Run the app:
```python
python main.py
```


# To-do
- Code better frontend.