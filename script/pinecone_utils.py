from pinecone import Pinecone
from openai import OpenAI
import os
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()

openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def get_dentsply_pinecone_index(pc: Pinecone, index_name: str = 'dentsply-mvp1-data') -> Pinecone.Index:
    """
    Get the Dentsply Pinecone index
    """
    return pc.Index(index_name)


def upload_all_embeddings_to_pinecone(pc_index: Pinecone.Index, embeddings: dict) -> None:
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
    # Break the embeddings into batches of 1000
    batch_size = 1000
    embedding_keys = list(embeddings.keys())
    for i in range(0, len(embedding_keys), batch_size):
        batch = {embedding_keys[j]: embeddings[embedding_keys[j]] for j in range(i, min(i + batch_size, len(embedding_keys)))}
        formatted_batch = [{"id": str(key), "values": value["embedding"]} for key, value in batch.items()]
        pc_index.upsert(formatted_batch)
        print(f"Uploaded batch {i+1} to {i+batch_size} to Pinecone")


def upload_all_embeddings_to_pinecone_with_metadata(pc_index: Pinecone.Index, embeddings: dict) -> None:
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
    # Break the embeddings into batches of 1000
    batch_size = 50
    embedding_keys = list(embeddings.keys())
    list_embedding_values = list(embeddings.values())
    embedding_values_chunk = [list_embedding_values[i]['Chunk'] for i in range(len(embedding_keys))]
    embedding_values_embedding = [list_embedding_values[i]['Embedding'] for i in range(len(embedding_keys))]

    for i in tqdm(range(0, len(embedding_keys), batch_size)):
        i_end = min(i + batch_size, len(embedding_keys))
        ids_batch = embedding_keys[i:i_end]
        vectors_batch = embedding_values_embedding[i:i_end]
        meta_batch = [{'text':line} for line in embedding_values_chunk[i:i_end]]
        to_upsert_list = list(zip(ids_batch, vectors_batch, meta_batch))
        pc_index.upsert(vectors=to_upsert_list)
        print(f"Uploaded batch {i+1} to {i+batch_size} to Pinecone")


def query_pinecone_index(pc_index: Pinecone.Index, query: str, top_k: int = 5) -> list:
    """
    Query the Pinecone index

    Args:
        pc_index (Pinecone.Index): Pinecone index
        query (str): Query string
        top_k (int): Number of results to return

    Returns:
        list: List of results
    """
    response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
        )

    # Extract the embedding vector
    embedding_vector = response.data[0].embedding
    print(type(embedding_vector))
    print(type(embedding_vector[0]))
    return pc_index.query(vector=embedding_vector, top_k=top_k, include_metadata=True)
        
