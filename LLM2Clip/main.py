import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, AutoConfig
from llm2vec import LLM2Vec

image_folder_path = './ziegler'
model_name_or_path = "microsoft/LLM2CLIP-Openai-L-14-336"
llm_model_name = 'microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned'

processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
model = AutoModel.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    trust_remote_code=True).to('cuda').eval()

config = AutoConfig.from_pretrained(
    llm_model_name, trust_remote_code=True
)
llm_model = AutoModel.from_pretrained(llm_model_name, torch_dtype=torch.float16, config=config, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model.config._name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'  # Workaround for LLM2VEC
l2v = LLM2Vec(llm_model, tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512)

def generate_image_embeddings(image_folder_path):
    image_embeddings = []
    image_files = os.listdir(image_folder_path)
    
    for image_file in image_files:
        image_path = os.path.join(image_folder_path, image_file)
        if image_path.endswith(('.jpg', '.jpeg', '.png')):
            image = Image.open(image_path)
            input_pixels = processor(images=image, return_tensors="pt").pixel_values.to('cuda')

            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.get_image_features(input_pixels)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_embeddings.append((image_features.cpu().numpy(), image_file))

    return image_embeddings

def search_top_images(query, image_embeddings, top_n=5):
    text_features = l2v.encode([query], convert_to_tensor=True).to('cuda')

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.get_text_features(text_features)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarities = []
        for image_features, image_file in image_embeddings:
            similarity = (100.0 * torch.from_numpy(image_features).to('cuda') @ text_features.T).item()
            similarities.append((similarity, image_file))

        similarities.sort(reverse=True, key=lambda x: x[0])
        return similarities[:top_n]

# Step 1: Generate embeddings for all images in the folder
print("Generating image embeddings...")
image_embeddings = generate_image_embeddings(image_folder_path)
print(f"Generated embeddings for {len(image_embeddings)} images.")

# Step 2: Get user query input and search for top similar images
user_query = input("Enter your query: ")

top_results = search_top_images(user_query, image_embeddings, top_n=5)

# Step 3: Display top 5 results
print("\nTop 5 matching images based on your query:")
for idx, (similarity, image_file) in enumerate(top_results):
    print(f"{idx + 1}. {image_file} with similarity score: {similarity:.4f}")