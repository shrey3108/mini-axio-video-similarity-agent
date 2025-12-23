# Mini Axio AI Agent – Video Similarity Finder

A simple AI-powered video similarity detection system that analyzes YouTube videos using visual and audio embeddings.

## Features

- 🎥 YouTube video download and processing
- 🖼️ Key frame extraction using OpenCV
- 🎵 Audio fingerprinting with MFCC
- 🤖 Visual embeddings using CLIP
- 📊 Similarity scoring (visual + audio)
- 🔍 Automatic classification (Re-upload, Edited copy, Related, Unrelated)
- 💡 AI-powered explanations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

1. Enter a YouTube video URL
2. Click "Scan Video"
3. Wait for processing (downloads video, extracts features)
4. View similarity results with classifications

## How It Works

1. **Download**: Uses yt-dlp to download the source video
2. **Frame Extraction**: Samples 6 keyframes uniformly from the video
3. **Audio Processing**: Extracts MFCC features from the first 30 seconds
4. **Visual Embeddings**: Generates CLIP embeddings for each frame
5. **Search**: Finds candidate videos on YouTube (optional API key)
6. **Comparison**: Computes cosine similarity for visual and audio
7. **Classification**: Rule-based AI agent classifies results
8. **Explanation**: Provides human-readable reasoning

## File Structure

- `app.py` - Streamlit UI and main application
- `utils.py` - All processing logic (video, audio, embeddings, similarity)
- `requirements.txt` - Python dependencies

## Optional: YouTube API Key

For real YouTube search, add your API key in the sidebar. Without it, the app uses mock data for demonstration.

## Notes

- First run will download CLIP model (~350MB)
- Processing time depends on video length
- Uses lower quality video for faster processing
