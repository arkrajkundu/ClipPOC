import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
from faiss_index_manager import retrieve_similar_videos

device = "cuda" if torch.cuda.is_available() else "cpu"

text_model = CLIPTextModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k").to(device).eval()
tokenizer = CLIPTokenizer.from_pretrained("Searchium-ai/clip4clip-webvid150k")

def get_text_embedding(query):
    """
    Converts a text query into an embedding using CLIP4CLIP.
    """
    with torch.no_grad():
        inputs = tokenizer(text=query, return_tensors="pt").to(device)
        outputs = text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        text_embedding = outputs[0] / outputs[0].norm(dim=-1, keepdim=True)
    return text_embedding.cpu().numpy()

def search_video_by_text():
    """
    Asks the user for a query and retrieves the most relevant video chunks.
    """
    query = input("Enter your search query: ")
    text_embedding = get_text_embedding(query)
    results = retrieve_similar_videos(text_embedding)

    print("\nSearch Results:")
    for idx, (video_chunk, distance) in enumerate(results):
        print(f"{idx+1}. {video_chunk} (Distance: {distance})")

search_video_by_text()