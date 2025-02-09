# CLIP4Clip Video Search POC

A proof-of-concept (POC) implementation of video search using **CLIP4Clip** and **FAISS**, enabling efficient video retrieval based on text queries.

## 📌 Features

- Extracts **video chunks** and processes them using `CLIP4Clip`.
- Stores video embeddings in a **FAISS** index for efficient similarity search.
- Supports **text-based search** to retrieve relevant video segments.

---

## ⚡ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/arkrajkundu/CLIP4Clip.git
cd CLIP4Clip
```

### 2️⃣ Install Dependencies
```bash
pip install torch transformers faiss-cpu numpy opencv-python pillow torchvision
```
> *Note:* If you have a CUDA-compatible GPU, install `faiss-gpu` instead of `faiss-cpu` for faster processing:
```bash
pip install faiss-gpu
```

---

## 🚀 How to Use

### **Step 1: Create and Manage FAISS Index**
Run the FAISS index setup to initialize the search system:
```bash
python faiss_index_manager.py
```

---

### **Step 2: Process Videos and Store Embeddings**
Extract video chunks, compute embeddings using `CLIP4Clip`, and store them in the FAISS index.
```bash
python video_embedding_extractor.py --video_path "your_video.mp4" --output_dir "video_chunks"
```
> This will generate **video chunks** and store their embeddings in FAISS.

---

### **Step 3: Search Videos Using Text**
Run the search script and enter a text query to retrieve the most relevant video chunks.
```bash
python text_to_video_search.py
```
Example Query:
```
Enter your search query: A person running in a park
```
> The script will return **top matching video chunks** along with similarity scores.

---

## 🔧 Configuration Options

| Argument         | Default            | Description                                |
|-----------------|--------------------|--------------------------------------------|
| `--video_path`  | `"your_video.mp4"`  | Path to the input video file.              |
| `--output_dir`  | `"video_chunks"`    | Directory to store extracted video chunks. |
| `--chunk_size`  | `150`               | Number of frames per chunk.                |
| `--top_k`       | `5`                 | Number of top matches to retrieve.         |

---

## 📂 Project Structure

```
📦 clip4clip-video-search
 ┣ 📜 faiss_index_manager.py        # FAISS index management
 ┣ 📜 video_embedding_extractor.py  # Extracts video embeddings using CLIP4Clip
 ┣ 📜 text_to_video_search.py       # Searches video using text input
 ┣ 📜 README.md                     # Documentation
 ┗ 📂 video_chunks/                 # Extracted video segments (generated)
```

---

## 📢 Contributions
Contributions are welcome! Feel free to submit **issues** or **pull requests**.

📧 Contact: [arkraj.oggy@example.com]