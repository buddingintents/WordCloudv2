import streamlit as st
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio
import os
import tempfile
from io import BytesIO
import fitz  # PyMuPDF for PDF processing
import nltk
from nltk.corpus import stopwords
import re
from collections import Counter
import time
import random
from mpl_toolkits.mplot3d import Axes3D
import math

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configure the page
st.set_page_config(
    page_title="3D PDF WordCloud Animator - Large Sphere",
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
.feature-highlight {
    background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_bytes = pdf_file.read()
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
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    try:
        nltk_stopwords = set(stopwords.words(language))
    except:
        nltk_stopwords = set(STOPWORDS)

    all_stopwords = nltk_stopwords.union(set(STOPWORDS))
    if custom_stopwords:
        custom_stop_set = set([word.strip().lower() for word in custom_stopwords.split(',')])
        all_stopwords = all_stopwords.union(custom_stop_set)

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

def fibonacci_sphere_points(n_points, randomize=True):
    """Generate points on a sphere using Fibonacci spiral"""
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append([x, y, z])

    if randomize:
        random.shuffle(points)

    return np.array(points)

def rotate_points_3d(points, angle_x, angle_y, angle_z):
    """Rotate 3D points using rotation matrices"""
    # Rotation matrices
    cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
    cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
    cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)

    # Rotation matrix around X axis
    R_x = np.array([[1, 0, 0],
                    [0, cos_x, -sin_x],
                    [0, sin_x, cos_x]])

    # Rotation matrix around Y axis
    R_y = np.array([[cos_y, 0, sin_y],
                    [0, 1, 0],
                    [-sin_y, 0, cos_y]])

    # Rotation matrix around Z axis
    R_z = np.array([[cos_z, -sin_z, 0],
                    [sin_z, cos_z, 0],
                    [0, 0, 1]])

    # Combined rotation matrix
    R = np.dot(R_z, np.dot(R_y, R_x))

    # Apply rotation
    rotated_points = np.dot(points, R.T)

    return rotated_points

def project_3d_to_2d(points_3d, width, height, viewer_distance=2.5, sphere_scale=0.85):
    """Project 3D points to 2D using perspective projection with enhanced scaling"""
    x_3d, y_3d, z_3d = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

    # Enhanced perspective projection with larger scaling
    scale_factor = min(width, height) * sphere_scale / 2

    # Perspective projection with reduced viewer distance for larger appearance
    perspective_scale = viewer_distance / (viewer_distance + z_3d)

    # Apply scaling and centering
    x_2d = (x_3d * perspective_scale * scale_factor) + width / 2
    y_2d = (y_3d * perspective_scale * scale_factor) + height / 2

    return x_2d, y_2d, z_3d

def get_depth_color_alpha(z_values, base_color, colormap):
    """Calculate color and alpha based on depth with enhanced contrast"""
    # Normalize z values to [0, 1] range with enhanced contrast
    z_normalized = (z_values - z_values.min()) / (z_values.max() - z_values.min())

    # Apply gamma correction for better visual contrast
    z_enhanced = np.power(z_normalized, 0.7)

    # Create colormap
    cmap = plt.get_cmap(colormap)
    colors = cmap(z_enhanced)

    # Enhanced alpha based on depth (more dramatic difference)
    alphas = 0.4 + 0.6 * z_enhanced

    return colors, alphas

def create_3d_rotating_wordcloud_frame(word_freq, sphere_points, rotation_angles, 
                                     width=800, height=600, colormap='viridis',
                                     background_color='black', sphere_scale=0.85):
    """Create a single frame of the 3D rotating word cloud with enhanced scaling"""
    try:
        # Rotate the sphere
        rotated_points = rotate_points_3d(sphere_points, 
                                        rotation_angles[0], 
                                        rotation_angles[1], 
                                        rotation_angles[2])

        # Project to 2D with enhanced scaling
        x_2d, y_2d, z_3d = project_3d_to_2d(rotated_points, width, height, 
                                           viewer_distance=2.5, sphere_scale=sphere_scale)

        # Create the plot
        fig, ax = plt.subplots(figsize=(width/100, height/100), 
                              facecolor=background_color)
        ax.set_facecolor(background_color)

        # Get colors and alphas based on depth
        colors, alphas = get_depth_color_alpha(z_3d, 'white', colormap)

        # Sort by depth (furthest first)
        depth_order = np.argsort(z_3d)

        words = list(word_freq.keys())
        max_freq = max(word_freq.values())
        min_freq = min(word_freq.values())

        for i, idx in enumerate(depth_order):
            if i >= len(words):
                break

            word = words[i]
            freq = word_freq[word]

            # Enhanced font size scaling with better size distribution
            # Frequency-based scaling (8-32 base range)
            freq_normalized = (freq - min_freq) / (max_freq - min_freq) if max_freq > min_freq else 0.5
            base_font_size = 12 + freq_normalized * 24

            # Depth-based scaling (0.6 to 1.4 multiplier)
            depth_factor = 0.6 + 0.8 * ((z_3d[idx] - z_3d.min()) / (z_3d.max() - z_3d.min()))

            # Final font size
            font_size = base_font_size * depth_factor

            # Ensure text stays within bounds
            if 0 <= x_2d[idx] <= width and 0 <= y_2d[idx] <= height:
                # Add text with enhanced styling
                ax.text(x_2d[idx], y_2d[idx], word,
                       fontsize=font_size,
                       ha='center', va='center',
                       color=colors[idx],
                       alpha=alphas[idx],
                       weight='bold',
                       family='sans-serif',
                       transform=ax.transData)

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.axis('off')
        plt.tight_layout(pad=0)

        # Convert to PIL Image
        buf = BytesIO()
        plt.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0, 
                   dpi=100, facecolor=background_color)
        buf.seek(0)

        image = Image.open(buf)
        plt.close(fig)

        return image
    except Exception as e:
        st.error(f"Error creating 3D frame: {str(e)}")
        return None

def create_3d_animated_wordcloud(text, color_scheme='viridis', num_frames=30, 
                                duration=200, width=800, height=600,
                                background_color='black', sphere_scale=0.85):
    """Create animated GIF of 3D rotating word cloud sphere with enhanced scaling"""
    frames = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Get word frequencies
    word_freq = Counter(text.split())
    top_words = dict(word_freq.most_common(50))  # Use top 50 words

    # Generate sphere points for words
    n_words = len(top_words)
    sphere_points = fibonacci_sphere_points(n_words)

    status_text.text('Generating large 3D rotating word cloud sphere...')

    for i in range(num_frames):
        progress = (i + 1) / num_frames
        progress_bar.progress(progress)
        status_text.text(f'Creating enhanced 3D frame {i+1}/{num_frames}...')

        # Calculate rotation angles for smooth animation
        angle_y = 2 * np.pi * i / num_frames  # Main rotation around Y axis
        angle_x = 0.2 * np.sin(2 * np.pi * i / num_frames)  # Slight wobble around X
        angle_z = 0.1 * np.cos(2 * np.pi * i / num_frames)  # Slight wobble around Z

        rotation_angles = (angle_x, angle_y, angle_z)

        frame = create_3d_rotating_wordcloud_frame(
            word_freq=top_words,
            sphere_points=sphere_points,
            rotation_angles=rotation_angles,
            width=width,
            height=height,
            colormap=color_scheme,
            background_color=background_color,
            sphere_scale=sphere_scale
        )

        if frame:
            frames.append(frame)

        time.sleep(0.05)

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

def create_static_3d_wordcloud(text, color_scheme='viridis', width=800, height=600,
                             background_color='black', rotation_angle=0, sphere_scale=0.85):
    """Create a static 3D word cloud with enhanced scaling"""
    word_freq = Counter(text.split())
    top_words = dict(word_freq.most_common(50))

    sphere_points = fibonacci_sphere_points(len(top_words))
    rotation_angles = (0.2, rotation_angle, 0.1)

    return create_3d_rotating_wordcloud_frame(
        word_freq=top_words,
        sphere_points=sphere_points,
        rotation_angles=rotation_angles,
        width=width,
        height=height,
        colormap=color_scheme,
        background_color=background_color,
        sphere_scale=sphere_scale
    )

def main():
    st.title("🌍 3D PDF WordCloud Sphere Animator - Large Scale")

    # Feature highlight
    st.markdown("""
    <div class="feature-highlight">
        🎯 <strong>ENHANCED!</strong> Large-scale 3D rotating sphere word clouds that fill the entire frame with maximum visual impact!
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Create stunning **large-scale 3D rotating sphere word clouds** from PDF documents! 
    The sphere now occupies most of the frame space for maximum visual impact, with words positioned 
    on a virtual sphere surface that rotates smoothly with realistic depth perception.
    """)

    # Sidebar configuration
    st.sidebar.title("⚙️ Enhanced 3D Configuration")

    # PDF Upload Section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "📄 Upload PDF File",
        type=["pdf"],
        help="Select a PDF file to extract text and create large-scale 3D word cloud sphere"
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
                "Language for Stopwords",
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

            # Enhanced 3D Visual configuration
            st.sidebar.subheader("🎨 Enhanced 3D Visual Settings")

            color_schemes = get_color_schemes()
            selected_color_scheme = st.sidebar.selectbox(
                "Color Theme",
                options=list(color_schemes.keys()),
                index=0
            )

            background_options = {
                'Black': 'black',
                'Dark Blue': '#001122',
                'Dark Purple': '#2d1b69',
                'Dark Green': '#0d2818',
                'Dark Gray': '#1a1a1a',
                'White': 'white'
            }

            selected_background = st.sidebar.selectbox(
                "Background Color",
                options=list(background_options.keys()),
                index=0
            )

            # Enhanced scaling control
            sphere_scale = st.sidebar.slider(
                "Sphere Size (Frame Coverage)",
                min_value=0.5, max_value=0.95, value=0.85, step=0.05,
                help="Controls how much of the frame the sphere occupies"
            )

            col1, col2 = st.sidebar.columns(2)
            with col1:
                width = st.number_input("Width", min_value=400, max_value=1200, value=800, step=50)
            with col2:
                height = st.number_input("Height", min_value=400, max_value=1200, value=600, step=50)

            # 3D Animation settings
            st.sidebar.subheader("🎬 Enhanced Animation Settings")

            col3, col4 = st.sidebar.columns(2)
            with col3:
                num_frames = st.slider("Rotation Frames", min_value=15, max_value=60, value=30)
            with col4:
                duration = st.slider("Frame Speed (ms)", min_value=100, max_value=500, value=200)

            rotation_speed = st.sidebar.slider("Rotation Speed", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

            # Generate buttons
            col_gen1, col_gen2 = st.columns(2)

            with col_gen1:
                if st.button("🖼️ Generate Large 3D WordCloud"):
                    with st.spinner("Creating large-scale 3D word cloud sphere..."):
                        static_wordcloud = create_static_3d_wordcloud(
                            text=processed_text,
                            color_scheme=color_schemes[selected_color_scheme],
                            width=width,
                            height=height,
                            background_color=background_options[selected_background],
                            rotation_angle=np.pi/4,
                            sphere_scale=sphere_scale
                        )

                        if static_wordcloud:
                            st.subheader("🌍 Large-Scale 3D WordCloud Sphere")
                            st.image(static_wordcloud, caption="Enhanced 3D Word Cloud Sphere", use_column_width=True)

                            # Download button for static image
                            buf = BytesIO()
                            static_wordcloud.save(buf, format='PNG')
                            buf.seek(0)

                            st.download_button(
                                label="💾 Download Large 3D Image",
                                data=buf.getvalue(),
                                file_name=f"large_3d_wordcloud_{uploaded_file.name.replace('.pdf', '')}.png",
                                mime="image/png"
                            )

            with col_gen2:
                if st.button("🎬 Generate Large Rotating Sphere"):
                    with st.spinner("Creating large-scale 3D rotating animation..."):
                        # Adjust frame count based on rotation speed
                        adjusted_frames = int(num_frames / rotation_speed)
                        adjusted_duration = int(duration * rotation_speed)

                        gif_buffer = create_3d_animated_wordcloud(
                            text=processed_text,
                            color_scheme=color_schemes[selected_color_scheme],
                            num_frames=adjusted_frames,
                            duration=adjusted_duration,
                            width=width,
                            height=height,
                            background_color=background_options[selected_background],
                            sphere_scale=sphere_scale
                        )

                        if gif_buffer:
                            st.subheader("🌍 Large-Scale 3D Rotating WordCloud Sphere")
                            st.image(gif_buffer.getvalue(), caption="Enhanced Large-Scale 3D Rotating Word Cloud Sphere")

                            # Download button for GIF
                            st.download_button(
                                label="💾 Download Large 3D Animated GIF",
                                data=gif_buffer.getvalue(),
                                file_name=f"large_3d_rotating_wordcloud_{uploaded_file.name.replace('.pdf', '')}.gif",
                                mime="image/gif"
                            )

                            st.success("🎉 Large-scale 3D rotating word cloud sphere created successfully!")
                        else:
                            st.error("Failed to create 3D animated GIF")
        else:
            st.error("❌ Failed to extract text from PDF. Please check if the file is valid.")

    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload a PDF file to get started!")

        # Sample features showcase
        st.subheader("✨ Enhanced Large-Scale Features")

        feature_cols = st.columns(3)

        with feature_cols[0]:
            st.markdown("""
            **🌍 Large-Scale Sphere**
            - Sphere covers up to 95% of frame
            - Enhanced perspective projection
            - Optimized scaling algorithms
            - Maximum visual impact
            """)

        with feature_cols[1]:
            st.markdown("""
            **🎨 Enhanced Visuals**
            - Improved depth contrast
            - Better font size distribution
            - Gamma-corrected colors
            - Professional quality output
            """)

        with feature_cols[2]:
            st.markdown("""
            **⚡ User Control**
            - Adjustable sphere size (50%-95%)
            - Custom frame coverage
            - Enhanced scaling options
            - Real-time size preview
            """)

        st.subheader("🎯 Large-Scale Enhancements")
        st.markdown("""
        1. **Enhanced Scaling**: Sphere can now occupy up to 95% of the frame space
        2. **Better Projection**: Improved 3D to 2D mapping for maximum utilization
        3. **Optimized Sizing**: Smart font scaling based on both frequency and depth
        4. **User Control**: Adjustable sphere size with real-time feedback
        5. **Professional Output**: Cinema-quality large-scale rendering
        """)

if __name__ == "__main__":
    main()
