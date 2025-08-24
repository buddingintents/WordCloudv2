import streamlit as st
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio
import os
import tempfile
import zipfile
from io import BytesIO
import fitz  # PyMuPDF for PDF processing
import nltk
from nltk.corpus import stopwords
import re
from collections import Counter
import time
import random

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configure the page
st.set_page_config(
    page_title="PDF WordCloud Animator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main {
    padding: 2rem;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 10px;
    font-weight: bold;
}
.upload-section {
    border: 2px dashed #667eea;
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        # Read PDF file from uploaded file
        pdf_bytes = pdf_file.read()

        # Open PDF with PyMuPDF
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()

        pdf_document.close()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def preprocess_text(text, custom_stopwords=None, language='english'):
    """Clean and preprocess text for word cloud generation"""
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Get stopwords
    try:
        nltk_stopwords = set(stopwords.words(language))
    except:
        nltk_stopwords = set(STOPWORDS)

    # Add custom stopwords
    all_stopwords = nltk_stopwords.union(set(STOPWORDS))
    if custom_stopwords:
        custom_stop_set = set([word.strip().lower() for word in custom_stopwords.split(',')])
        all_stopwords = all_stopwords.union(custom_stop_set)

    # Tokenize and remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in all_stopwords and len(word) > 2]

    return ' '.join(filtered_words)

def get_color_schemes():
    """Return available color schemes for word clouds"""
    return {
        'Viridis': 'viridis',
        'Plasma': 'plasma',
        'Inferno': 'inferno',
        'Magma': 'magma',
        'Ocean Blue': 'Blues',
        'Forest Green': 'Greens',
        'Sunset Orange': 'Oranges',
        'Purple Rain': 'Purples',
        'Fire Red': 'Reds',
        'Cool Winter': 'winter',
        'Hot Summer': 'hot',
        'Rainbow': 'rainbow',
        'Autumn Leaves': 'autumn',
        'Spring Fresh': 'spring'
    }

def create_wordcloud_frame(text, width=800, height=400, colormap='viridis', 
                          background_color='white', max_words=100, 
                          relative_scaling=0.5, font_step=1):
    """Create a single word cloud frame"""
    try:
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            colormap=colormap,
            max_words=max_words,
            relative_scaling=relative_scaling,
            font_step=font_step,
            prefer_horizontal=0.7,
            min_font_size=10,
            scale=2
        ).generate(text)

        # Convert to PIL Image
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)

        # Save to bytes
        buf = BytesIO()
        plt.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0, dpi=100)
        buf.seek(0)

        # Convert to PIL Image
        image = Image.open(buf)
        plt.close(fig)

        return image
    except Exception as e:
        st.error(f"Error creating word cloud: {str(e)}")
        return None

def create_animated_wordcloud(text, color_scheme='viridis', num_frames=20, 
                            duration=500, width=800, height=400):
    """Create animated GIF of word cloud with varying parameters"""
    frames = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Get word frequencies
    word_freq = Counter(text.split())

    for i in range(num_frames):
        progress = (i + 1) / num_frames
        progress_bar.progress(progress)
        status_text.text(f'Generating frame {i+1}/{num_frames}...')

        # Vary parameters for animation effect
        max_words = int(50 + (i / num_frames) * 100)  # Gradually increase words
        relative_scaling = 0.3 + (i / num_frames) * 0.4  # Vary scaling

        # Create slight variations in color intensity
        if color_scheme in ['viridis', 'plasma', 'inferno', 'magma']:
            current_colormap = color_scheme
        else:
            current_colormap = color_scheme

        frame = create_wordcloud_frame(
            text=text,
            width=width,
            height=height,
            colormap=current_colormap,
            max_words=max_words,
            relative_scaling=relative_scaling,
            font_step=1 + i % 3
        )

        if frame:
            frames.append(frame)

        time.sleep(0.1)  # Small delay for visual feedback

    progress_bar.empty()
    status_text.empty()

    if frames:
        # Create GIF
        gif_buffer = BytesIO()
        frames[0].save(
            gif_buffer,
            format='GIF',
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
            optimize=True
        )
        gif_buffer.seek(0)
        return gif_buffer

    return None

def main():
    st.title("📚 PDF WordCloud Animator")
    st.markdown("""
    Create beautiful, animated word clouds from PDF documents! 
    Upload a PDF, customize the appearance, and generate an animated GIF showing word frequency.
    """)

    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")

    # PDF Upload Section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "📄 Upload PDF File",
        type=["pdf"],
        help="Select a PDF file to extract text and create word cloud"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # Show file details
        st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
        st.sidebar.write(f"📊 File size: {uploaded_file.size / 1024:.1f} KB")

        # Extract text
        with st.spinner("Extracting text from PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)

        if extracted_text:
            # Show text preview
            with st.expander("📖 Preview Extracted Text"):
                st.text_area("Text Preview", extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text, height=200)

            # Text preprocessing options
            st.sidebar.subheader("📝 Text Preprocessing")

            language_options = {
                'English': 'english',
                'Spanish': 'spanish',
                'French': 'french',
                'German': 'german',
                'Portuguese': 'portuguese',
                'Italian': 'italian'
            }

            selected_language = st.sidebar.selectbox(
                "Select Language for Stopwords",
                options=list(language_options.keys()),
                index=0
            )

            custom_stopwords = st.sidebar.text_area(
                "Custom Stopwords (comma-separated)",
                help="Add custom words to exclude from the word cloud",
                placeholder="word1, word2, word3"
            )

            # Preprocess text
            processed_text = preprocess_text(
                extracted_text, 
                custom_stopwords, 
                language_options[selected_language]
            )

            # Word cloud configuration
            st.sidebar.subheader("🎨 Visual Settings")

            color_schemes = get_color_schemes()
            selected_color_scheme = st.sidebar.selectbox(
                "Color Theme",
                options=list(color_schemes.keys()),
                index=0
            )

            col1, col2 = st.sidebar.columns(2)
            with col1:
                width = st.number_input("Width", min_value=400, max_value=1200, value=800, step=50)
            with col2:
                height = st.number_input("Height", min_value=300, max_value=800, value=400, step=50)

            # Animation settings
            st.sidebar.subheader("🎬 Animation Settings")

            col3, col4 = st.sidebar.columns(2)
            with col3:
                num_frames = st.slider("Number of Frames", min_value=5, max_value=30, value=15)
            with col4:
                duration = st.slider("Frame Duration (ms)", min_value=200, max_value=1000, value=400)

            # Generate buttons
            col_gen1, col_gen2 = st.columns(2)

            with col_gen1:
                if st.button("🖼️ Generate Static WordCloud"):
                    with st.spinner("Creating word cloud..."):
                        static_wordcloud = create_wordcloud_frame(
                            text=processed_text,
                            width=width,
                            height=height,
                            colormap=color_schemes[selected_color_scheme],
                            max_words=150
                        )

                        if static_wordcloud:
                            st.subheader("📊 Generated WordCloud")
                            st.image(static_wordcloud, caption="Static Word Cloud", use_column_width=True)

                            # Download button for static image
                            buf = BytesIO()
                            static_wordcloud.save(buf, format='PNG')
                            buf.seek(0)

                            st.download_button(
                                label="💾 Download Static Image",
                                data=buf.getvalue(),
                                file_name=f"wordcloud_{uploaded_file.name.replace('.pdf', '')}.png",
                                mime="image/png"
                            )

            with col_gen2:
                if st.button("🎬 Generate Animated GIF"):
                    with st.spinner("Creating animated word cloud..."):
                        gif_buffer = create_animated_wordcloud(
                            text=processed_text,
                            color_scheme=color_schemes[selected_color_scheme],
                            num_frames=num_frames,
                            duration=duration,
                            width=width,
                            height=height
                        )

                        if gif_buffer:
                            st.subheader("🎥 Animated WordCloud")
                            st.image(gif_buffer.getvalue(), caption="Animated Word Cloud GIF")

                            # Download button for GIF
                            st.download_button(
                                label="💾 Download Animated GIF",
                                data=gif_buffer.getvalue(),
                                file_name=f"animated_wordcloud_{uploaded_file.name.replace('.pdf', '')}.gif",
                                mime="image/gif"
                            )

                            st.success("🎉 Animated GIF created successfully!")
                        else:
                            st.error("Failed to create animated GIF")
        else:
            st.error("❌ Failed to extract text from PDF. Please check if the file is valid.")

    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload a PDF file to get started!")

        # Sample features showcase
        st.subheader("✨ Features")

        feature_cols = st.columns(3)

        with feature_cols[0]:
            st.markdown("""
            **📄 PDF Processing**
            - Extract text from any PDF
            - Support for multi-page documents
            - Robust text extraction
            """)

        with feature_cols[1]:
            st.markdown("""
            **🎨 Customization**
            - Multiple color themes
            - Adjustable dimensions
            - Custom stopwords
            """)

        with feature_cols[2]:
            st.markdown("""
            **🎬 Animation**
            - Smooth GIF generation
            - Configurable frame rate
            - Professional quality output
            """)

if __name__ == "__main__":
    main()
