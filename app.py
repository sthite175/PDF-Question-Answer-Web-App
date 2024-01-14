
#-------------------------IMPORT ALL LIBRARY-------------------------------------------------------

import streamlit as st
import os
import numpy as np 
import pandas as pd
import pinecone
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceHubEmbeddings #, OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Pinecone, FAISS
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

# Embedding Technique of OpenAI
KEY = os.environ['OPENAI_API_KEY']
embedding = OpenAIEmbeddings(api_key=KEY)

#----------------------------CREATE STREAMLIT WEB APP------------------------------------------------
st.title("📝 PDF Question-Answer Web App")

st.sidebar.title("Upload PDF File Here..")

# Get user input: PDF file
pdf_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])


filepath = "faiss_store"
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)
import tempfile
from tempfile import NamedTemporaryFile
temp_dir = tempfile.mkdtemp()

#---------------------------------------READ PDF FILE---------------------------------------------------
if pdf_file:
    # Save PDF content to a temporary file
    with NamedTemporaryFile(delete=False) as temp:
        temp.write(pdf_file.read())
        documents = PyPDFLoader(temp.name).load_and_split()

    # Load PDF File
    #file_loader = PyPDFLoader(temp_pdf_file)
    #documents = file_loader.load_and_split()
    
    # Text Splitter
    text_splittter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=10)
    text_chunks = text_splittter.split_documents(documents)

    # EMBEDDING AND SAVING DATA TO FAISS INDEX
    vectorindex_openai = FAISS.from_documents(text_chunks, embedding)

    # SAVE  FILE
    vectorindex_openai.save_local(filepath)


query = st.text_input("Questions: ",placeholder="Ask any question related to this file")
#----------------------QUERY---------------------------------------------------------------------------
if query:
    if os.path.exists(filepath):
        # Load Faiss Folder
        vector_file = FAISS.load_local(filepath, embedding)
        chain = load_qa_chain(llm=llm, chain_type='stuff')
        docs = vector_file.similarity_search(query)

        answer = chain.run(input_documents=docs, question=query)
        # Display the answer
        st.subheader("Answer:")
        st.write(answer)

#-----------------------------------------END----------------------------------------------------------