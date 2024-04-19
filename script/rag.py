from dotenv import load_dotenv
import PyPDF2
import sqlite3
import numpy as np
import pickle
from openai import OpenAI
import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm


def extract_text_from_pdf(pdf_file):
    '''
    This function extracts text from a PDF file.
    '''
    pdf_file = open(pdf_file, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

def chunking(text):
    '''
    This function chunks the text into smaller pieces to be used for creating embeddings.
    Chunk size is 1000 and the overlap is 200.
    '''
    chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
    return chunks

def make_embeddings(client, chunks):
    '''
    This function creates embeddings for the chunks of text using the OpenAI API.
    '''
    
    def _make_embedding(client, chunk, model="text-embedding-3-small"):
        chunk = chunk.replace("\n", " ")
        return client.embeddings.create(input = [chunk], model=model).data[0].embedding
    
    embeddings = []
    for chunk in chunks:
        embedding = _make_embedding(client, chunk)
        embeddings.append(embedding)
    return embeddings

def create_database():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index('aipi590-project2')
    return index

def upload_to_database(index, embeddings):
    # embedding_keys = list(embeddings.keys())
    
    batch_size = 50
    embedding_keys = list(embeddings.keys())
    list_embedding_values = list(embeddings.values())
    embedding_values_chunk = [list_embedding_values[i]['text'] for i in range(len(embedding_keys))]
    embedding_values_embedding = [list_embedding_values[i]['embedding'] for i in range(len(embedding_keys))]

    for i in tqdm(range(0, len(embedding_keys), batch_size)):
        i_end = min(i + batch_size, len(embedding_keys))
        ids_batch = embedding_keys[i:i_end]
        vectors_batch = embedding_values_embedding[i:i_end]
        meta_batch = [{'text':line} for line in embedding_values_chunk[i:i_end]]
        to_upsert_list = list(zip(ids_batch, vectors_batch, meta_batch))
        index.upsert(vectors=to_upsert_list)
        print(f"Uploaded batch {i+1} to {i+batch_size} to Pinecone")


def main():
    load_dotenv(override=True)
    
    openai_key = os.getenv("OPENAI_KEY")
    
    client = OpenAI(api_key=openai_key)
    
    data_path = "../data/"
    meta_chunks = []; meta_embedding = []
    meta_embeddings = {}
    # iterate files 
    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            file = os.path.join(data_path, file)
            
            text = extract_text_from_pdf(file)
            chunks = chunking(text)
            embeddings = make_embeddings(client, chunks)
            
            meta_chunks = meta_chunks + chunks
            meta_embedding = meta_embedding + embeddings

    for idx in range(len(meta_chunks)):
        meta_embeddings[str(idx)] = {"text": meta_chunks[idx], 
                                "embedding": meta_embedding[idx]}
    index = create_database()
    upload_to_database(index, meta_embeddings)
    pass


if __name__ == "__main__":
    main()

    
    