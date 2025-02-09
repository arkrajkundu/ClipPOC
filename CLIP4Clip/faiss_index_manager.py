import faiss
import numpy as np
import os

FAISS_INDEX_PATH = "./video_index.faiss"
MAPPING_FILE = "./chunk_mapping.txt"

print("Script 1 initializing...")

def create_faiss_index(dimension=512):
    """
    Creates and initializes a FAISS index for L2 distance search.
    """
    print("Creating FAISS index...")
    index = faiss.IndexFlatL2(dimension)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index created and saved at {FAISS_INDEX_PATH}")

def insert_embeddings_to_faiss(embeddings, chunk_files):
    """
    Inserts new embeddings into the FAISS database and updates chunk mapping.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        create_faiss_index(embeddings.shape[1])

    index = faiss.read_index(FAISS_INDEX_PATH)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(MAPPING_FILE, "a") as f:
        for i, chunk in enumerate(chunk_files):
            f.write(f"{index.ntotal - len(chunk_files) + i},{chunk}\n")

    print(f"Inserted {len(embeddings)} embeddings into FAISS.")

def retrieve_similar_videos(query_embedding, top_k=5):
    """
    Retrieves the most similar video chunks based on query embedding.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        print("FAISS index does not exist.")
        return []

    index = faiss.read_index(FAISS_INDEX_PATH)
    distances, indices = index.search(query_embedding, top_k)

    with open(MAPPING_FILE, "r") as f:
        chunk_mapping = {int(line.split(",")[0]): line.split(",")[1].strip() for line in f.readlines()}

    results = [(chunk_mapping[idx], dist) for idx, dist in zip(indices[0], distances[0])]
    
    print("\nTop Matching Video Chunks:")
    for chunk, dist in results:
        print(f"Video: {chunk}, Distance: {dist}")

    return results

if __name__ == "__main__":
    if not os.path.exists(FAISS_INDEX_PATH):
        print("FAISS index not found. Creating a new one...")
        create_faiss_index()
    else:
        print("FAISS index already exists!")