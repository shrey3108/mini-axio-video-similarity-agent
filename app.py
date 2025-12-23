"""
Mini Axio AI Agent – Video Similarity Finder
Streamlit UI for video similarity detection
"""

import streamlit as st
from utils import SimilarityDetector
import pandas as pd


# Page config
st.set_page_config(
    page_title="Video Similarity Finder",
    page_icon="🎥",
    layout="wide"
)

# Title and description
st.title("🎥 Mini Axio AI Agent – Video Similarity Finder")
st.markdown("Upload a YouTube video URL to find similar videos using AI-powered visual and audio analysis.")

# Sidebar for optional API key
with st.sidebar:
    
    
    st.markdown("---")
    st.markdown("""
    ### How it works:
    1. **Download** video from YouTube
    2. **Extract** key frames & audio
    3. **Generate** embeddings (CLIP)
    4. **Search** for similar videos (yt-dlp)
    5. **Compare** & classify results
    """)

# Main input section
col1, col2 = st.columns([3, 1])

with col1:
    video_url = st.text_input(
        "Enter YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste the full YouTube video URL"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)  # Spacing
    scan_button = st.button("🔍 Scan Video", type="primary", use_container_width=True)

# Processing logic
if scan_button:
    if not video_url:
        st.error("⚠️ Please enter a YouTube URL")
    elif "youtube.com" not in video_url and "youtu.be" not in video_url:
        st.error("⚠️ Please enter a valid YouTube URL")
    else:
        # Initialize detector (no API key needed)
        detector = SimilarityDetector()
        
        # Processing with status updates
        with st.spinner("🎬 Processing video... This may take a minute..."):
            try:
                # Run the pipeline (no API key needed)
                results = detector.process_and_compare(video_url)
                
                # Store in session state
                st.session_state['results'] = results
                st.session_state['source_url'] = video_url
                
            except Exception as e:
                st.error(f"❌ Error processing video: {str(e)}")
                st.stop()
        
        st.success("✅ Analysis complete!")

# Display results
if 'results' in st.session_state and st.session_state['results']:
    st.markdown("---")
    st.header("📊 Similarity Results")
    
    results = st.session_state['results']
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Videos Analyzed", len(results))
    with col2:
        reuploads = sum(1 for r in results if r['classification'] == 'Re-upload')
        st.metric("Potential Re-uploads", reuploads)
    with col3:
        avg_similarity = sum(r['similarity'] for r in results) / len(results) if results else 0
        st.metric("Avg Similarity", f"{avg_similarity:.1f}%")
    
    st.markdown("---")
    
    # Display each result as a card
    for idx, result in enumerate(results, 1):
        # Color code by classification
        if result['classification'] == 'Re-upload':
            badge_color = "🔴"
        elif result['classification'] == 'Edited copy':
            badge_color = "🟡"
        elif result['classification'] == 'Related content':
            badge_color = "🟢"
        else:
            badge_color = "⚪"
        
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### {badge_color} Result {idx}: {result['title']}")
                st.markdown(f"**URL:** [{result['url']}]({result['url']})")
                st.markdown(f"**Classification:** `{result['classification']}`")
                st.markdown(f"**Explanation:** {result['reason']}")
            
            with col2:
                st.metric("Overall Similarity", f"{result['similarity']}%")
                st.caption(f"Visual: {result['visual_similarity']}%")
                st.caption(f"Audio: {result['audio_similarity']}%")
        
        st.markdown("---")
    
    # Export option
    if st.button("📥 Export Results as CSV"):
        df = pd.DataFrame(results)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="similarity_results.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Video processing powered by CLIP, OpenCV, and Librosa")
