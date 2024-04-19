from dotenv import load_dotenv
import PyPDF2
import numpy as np
from openai import OpenAI
import os
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from bs4 import BeautifulSoup
import requests


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
    """
    Upload all embeddings to Pinecone

    Args:
        pc_index (Pinecone.Index): Pinecone index
        embeddings (dict): Dictionary of embeddings
            The embedding dict should be of this format:
                {0: 
                    {
                    "embedding": [0.1, 0.2, ..., 0.9],
                    "text": "......"
                    },
                1: 
                    {...},
                }
    """
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


def make_meta_embeddings_html(client):
    url_list = [
        "https://ai.meng.duke.edu/",
        "https://pratt.duke.edu/life/resources/grad/",
        "https://sites.duke.edu/prattgsps/engineering-masters-programs-student-advisory-board/",
        
        
    ]
    meta_chunks = []; meta_embedding = []
    meta_embeddings = {}
    def _extract_text_from(url):
        html = requests.get(url).text
        soup = BeautifulSoup(html, features="html.parser")
        text = soup.get_text()

        lines = (line.strip() for line in text.splitlines())
        return '\n'.join(line for line in lines if line)

    for url in url_list:
        text = _extract_text_from(url)
        chunks = chunking(text)
        embeddings = make_embeddings(client, chunks)
        
        meta_chunks = meta_chunks + chunks
        meta_embedding = meta_embedding + embeddings
        
    for idx in range(len(meta_chunks)):
        meta_embeddings[str(idx)] = {"text": meta_chunks[idx], 
                                "embedding": meta_embedding[idx]}
    return meta_embeddings

        
def make_meta_embeddings_pdf(client):
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
    return meta_embeddings

def search_similar_text(index, query_embedding, top_k=5):    
    context = ""
    
    result = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_values=True,
        include_metadata=True
        )
    # get the top 5 similar texts
    for idx in range(top_k):
        context = context + result['matches'][idx]['metadata']['text']
        
    return context


def main():
    load_dotenv(override=True)
    
    openai_key = os.getenv("OPENAI_KEY")
    
    client = OpenAI(api_key=openai_key)
    
    index = create_database()
    
    # meta_embeddings_pdf = make_meta_embeddings_pdf(client)
    meta_embeddings_html = make_meta_embeddings_html(client)
    
    # print(len(meta_embeddings_html))
    # print(len(meta_embeddings_pdf))
    
    # upload_to_database(index, meta_embeddings_pdf)
    upload_to_database(index, meta_embeddings_html)
    
    # query = "what course should i choose in the first semester of the AI meng program at Duke?"
    
    # query_embedding = make_embeddings(client, [query])[0]
    # context = search_similar_text(index, query_embedding)
    
    
    
    pass


if __name__ == "__main__":
    main()

    
    