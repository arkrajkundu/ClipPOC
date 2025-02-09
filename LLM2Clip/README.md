# LLM2CLIP Image Search POC

A proof-of-concept (POC) implementation of **text-to-image retrieval** using **LLM2CLIP**. This project enables efficient **semantic image search** by extracting **image embeddings** and matching them with user queries.

## 📌 Features

- Extracts **frames from videos** and processes them as individual images.
- Computes **image embeddings** using `LLM2CLIP`.
- Converts **text queries** into embeddings for similarity search.
- Retrieves the **most relevant images** based on text input.

---

## ⚡ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/arkrajkundu/LLM2Clip.git
cd LLM2Clip
```

### 2️⃣ Install Dependencies
```bash
pip install torch transformers opencv-python pillow numpy llm2vec
```
> *Note:* If you have a **CUDA-compatible GPU**, install `torch` with GPU support for better performance.

---

## 🚀 How to Use

### **Step 1: Extract Frames from a Video**
Extract frames from a video at regular intervals and save them as images.
```bash
python extract_frames.py --video_path "your_video.mp4" --frame_interval 20 --output_dir "extracted_frames"
```
> This will create a folder with extracted **video frames**.

---

### **Step 2: Generate Image Embeddings**
Run the script to compute **image embeddings** for all extracted frames.
```bash
python main.py
```
> The script will process the images and store their embeddings.

---

### **Step 3: Search for Similar Images using Text**
After generating embeddings, you can **search for images** using a **text query**.
```bash
Enter your query: A person riding a bicycle
```
> The script will return **top 5 matching images** based on similarity scores.

---

## 🔧 Configuration Options

| Argument         | Default            | Description                                |
|-----------------|--------------------|--------------------------------------------|
| `--video_path`  | `"your_video.mp4"`  | Path to the input video file.              |
| `--frame_interval` | `20`            | Extract every Nth frame from the video.   |
| `--output_dir`  | `"extracted_frames"` | Directory to store extracted frames.      |
| `--top_n`       | `5`                | Number of top matches to retrieve during search. |

---

## 📂 Project Structure

```
📦 LLM2Clip
 ┣ 📜 main.py                 # Image embedding and search system
 ┣ 📜 extract_frames.py        # Extracts frames from videos
 ┣ 📜 README.md                # Documentation
 ┗ 📂 extracted_frames/        # Extracted frames from videos (generated)
```

---

## 📢 Contributions
Contributions are welcome! Feel free to submit **issues** or **pull requests**.

📧 Contact: [arkraj.oggy@example.com]