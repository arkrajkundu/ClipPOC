import torch
import cv2
import numpy as np
import os
from PIL import Image
from transformers import CLIPVisionModelWithProjection
from faiss_index_manager import insert_embeddings_to_faiss
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

device = "cuda" if torch.cuda.is_available() else "cpu"
print("CUDA: ", torch.cuda.is_available())

model = CLIPVisionModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k").to(device).eval()

def preprocess_frame(size=224):
    return Compose([
        Resize(size, interpolation=InterpolationMode.BICUBIC),            
        CenterCrop(size),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def extract_video_chunks(video_path, chunk_size=150, output_dir="video_chunks"):
    """
    Extracts video chunks of fixed size and saves them using OpenCV.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    chunks = []
    chunk_files = []
    chunk_index = 0

    for start_frame in range(0, total_frames, chunk_size):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []

        chunk_path = os.path.join(output_dir, f"chunk_{chunk_index}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(chunk_path, fourcc, frame_rate, (frame_width, frame_height))

        for _ in range(chunk_size):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = preprocess_frame()(Image.fromarray(frame)).unsqueeze(0)
            frames.append(processed_frame)

        out.release()
        if frames:
            video_chunk = torch.cat(frames).to(device)
            chunks.append(video_chunk)
            chunk_files.append(chunk_path)
            chunk_index += 1

    cap.release()
    print(f"Extracted {len(chunk_files)} video chunks.")
    return chunks, chunk_files

def extract_video_embedding(video_chunks):
    """
    Extracts a single video embedding per chunk using CLIP4CLIP.
    """
    video_embeddings = []

    for video_chunk in video_chunks:
        with torch.no_grad():
            visual_output = model(video_chunk)["image_embeds"]
            visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
            chunk_embedding = torch.mean(visual_output, dim=0)
            chunk_embedding = chunk_embedding / chunk_embedding.norm(dim=-1, keepdim=True)
            video_embeddings.append(chunk_embedding.cpu().numpy())

    return np.array(video_embeddings)

video_path = "<VIDEO_PATH>"
video_chunks, chunk_files = extract_video_chunks(video_path, chunk_size=150, output_dir="your_video_chunks")
video_embeddings = extract_video_embedding(video_chunks)

if len(video_embeddings) > 0:
    insert_embeddings_to_faiss(video_embeddings, chunk_files)