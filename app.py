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
    page_title="3D PDF WordCloud - Dynamic Word Count",
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
.word-stats {
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
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

def analyze_word_statistics(text):
    """Analyze word statistics from preprocessed text"""
    if not text:
        return {}, 0, 0

    word_freq = Counter(text.split())
    total_words = sum(word_freq.values())
    unique_words = len(word_freq)

    return word_freq, total_words, unique_words

def calculate_word_count_range(unique_words):
    """Calculate the word count range (60-100% of unique words)"""
    min_words = max(10, int(unique_words * 0.6))  # At least 10 words, or 60%
    max_words = unique_words

    # Reasonable limits for visualization
    min_words = min(min_words, 200)  # Cap at 200 for performance
    max_words = min(max_words, 300)  # Cap at 300 for performance

    return min_words, max_words

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

def create_word_position_mapping(word_freq, sphere_points):
    """Create a proper mapping between words and their sphere positions"""
    words = list(word_freq.keys())
    n_words = len(words)
    n_points = len(sphere_points)

    # Ensure we have enough points for words
    if n_points < n_words:
        # If not enough points, take only the top words
        words = words[:n_points]
    elif n_points > n_words:
        # If too many points, use only the first n_words points
        sphere_points = sphere_points[:n_words]

    # Create word-position mapping
    word_positions = {}
    for i, word in enumerate(words):
        word_positions[word] = {
            'position': sphere_points[i],
            'frequency': word_freq[word],
            'index': i
        }

    return word_positions

def adjust_font_sizes_for_word_count(base_min_size, base_max_size, word_count):
    """Adjust font size range based on number of words to maintain readability"""
    if word_count <= 50:
        # Fewer words - can use larger sizes
        return base_min_size + 2, base_max_size + 5
    elif word_count <= 100:
        # Medium word count - normal sizes
        return base_min_size, base_max_size
    elif word_count <= 150:
        # More words - slightly smaller sizes
        return base_min_size - 2, base_max_size - 3
    else:
        # Many words - smaller sizes to fit more
        return max(8, base_min_size - 4), max(25, base_max_size - 8)

def create_3d_rotating_wordcloud_frame(word_freq, sphere_points, rotation_angles, 
                                     width=800, height=600, colormap='viridis',
                                     background_color='black', sphere_scale=0.85):
    """Create a single frame of the 3D rotating word cloud with dynamic word count"""
    try:
        # Create word-position mapping
        word_positions = create_word_position_mapping(word_freq, sphere_points)

        if not word_positions:
            return None

        # Get all positions for rotation
        positions_array = np.array([data['position'] for data in word_positions.values()])

        # Rotate the sphere
        rotated_points = rotate_points_3d(positions_array, 
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

        # Get frequency range for proper scaling
        frequencies = [data['frequency'] for data in word_positions.values()]
        max_freq = max(frequencies)
        min_freq = min(frequencies)
        freq_range = max_freq - min_freq if max_freq > min_freq else 1

        # Adjust font sizes based on word count
        word_count = len(word_positions)
        min_font_size, max_font_size = adjust_font_sizes_for_word_count(10, 40, word_count)
        font_range = max_font_size - min_font_size

        # Sort by depth (furthest first) for proper rendering
        depth_order = np.argsort(z_3d)

        words_list = list(word_positions.keys())

        for render_idx in depth_order:
            if render_idx >= len(words_list):
                continue

            word = words_list[render_idx]
            word_data = word_positions[word]
            freq = word_data['frequency']

            # Calculate font size based on ACTUAL word frequency
            # Normalize frequency to 0-1 range
            freq_normalized = (freq - min_freq) / freq_range if freq_range > 0 else 0.5

            # Base font size using adjusted range
            base_font_size = min_font_size + freq_normalized * font_range

            # Depth-based scaling (0.7 to 1.3 multiplier)
            depth_normalized = (z_3d[render_idx] - z_3d.min()) / (z_3d.max() - z_3d.min())
            depth_factor = 0.7 + 0.6 * depth_normalized

            # Final font size
            final_font_size = base_font_size * depth_factor

            # Ensure minimum readable size
            final_font_size = max(final_font_size, 6)

            # Position text
            x_pos = x_2d[render_idx]
            y_pos = y_2d[render_idx]

            # Only render if within bounds
            if 0 <= x_pos <= width and 0 <= y_pos <= height:
                # Add text with frequency-appropriate styling
                ax.text(x_pos, y_pos, word,
                       fontsize=final_font_size,
                       ha='center', va='center',
                       color=colors[render_idx],
                       alpha=alphas[render_idx],
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

def create_3d_animated_wordcloud(text, selected_word_count, color_scheme='viridis', num_frames=30, 
                                duration=200, width=800, height=600,
                                background_color='black', sphere_scale=0.85):
    """Create animated GIF of 3D rotating word cloud sphere with dynamic word count"""
    frames = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Get word frequencies and select the specified number of words
    word_freq = Counter(text.split())
    selected_words = dict(word_freq.most_common(selected_word_count))

    # Generate sphere points for selected words
    n_words = len(selected_words)
    sphere_points = fibonacci_sphere_points(n_words)

    status_text.text(f'Generating 3D rotating word cloud with {n_words} words...')

    for i in range(num_frames):
        progress = (i + 1) / num_frames
        progress_bar.progress(progress)
        status_text.text(f'Creating frame {i+1}/{num_frames} with {n_words} words...')

        # Calculate rotation angles for smooth animation
        angle_y = 2 * np.pi * i / num_frames  # Main rotation around Y axis
        angle_x = 0.2 * np.sin(2 * np.pi * i / num_frames)  # Slight wobble around X
        angle_z = 0.1 * np.cos(2 * np.pi * i / num_frames)  # Slight wobble around Z

        rotation_angles = (angle_x, angle_y, angle_z)

        frame = create_3d_rotating_wordcloud_frame(
            word_freq=selected_words,
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

def create_static_3d_wordcloud(text, selected_word_count, color_scheme='viridis', width=800, height=600,
                             background_color='black', rotation_angle=0, sphere_scale=0.85):
    """Create a static 3D word cloud with dynamic word count"""
    word_freq = Counter(text.split())
    selected_words = dict(word_freq.most_common(selected_word_count))

    sphere_points = fibonacci_sphere_points(len(selected_words))
    rotation_angles = (0.2, rotation_angle, 0.1)

    return create_3d_rotating_wordcloud_frame(
        word_freq=selected_words,
        sphere_points=sphere_points,
        rotation_angles=rotation_angles,
        width=width,
        height=height,
        colormap=color_scheme,
        background_color=background_color,
        sphere_scale=sphere_scale
    )

def main():
    st.title("🌍 3D PDF WordCloud - Dynamic Word Count Control")

    # Feature highlight
    st.markdown("""
    <div class="feature-highlight">
        🎯 <strong>NEW FEATURE!</strong> Control the number of words in your wordcloud - choose 60-100% of available unique words!
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Create stunning **3D rotating sphere word clouds** with **customizable word count** from PDF documents! 
    Choose how many words to include (60-100% of unique words) for optimal visualization density and detail.
    """)

    # Sidebar configuration
    st.sidebar.title("⚙️ Dynamic Word Count Configuration")

    # PDF Upload Section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "📄 Upload PDF File",
        type=["pdf"],
        help="Select a PDF file to analyze and create dynamic word count wordcloud"
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

            # Analyze word statistics
            word_freq, total_words, unique_words = analyze_word_statistics(processed_text)

            if unique_words > 0:
                # Calculate word count range
                min_words, max_words = calculate_word_count_range(unique_words)

                # Display word statistics
                st.markdown("""
                <div class="word-stats">
                    📊 <strong>Document Word Analysis</strong><br>
                    • Total words (after preprocessing): {:,}<br>
                    • Unique words available: {:,}<br>
                    • Word count range: {:,} - {:,} words (60-100%)
                </div>
                """.format(total_words, unique_words, min_words, max_words), unsafe_allow_html=True)

                # Word count selection
                st.sidebar.subheader("📈 Dynamic Word Count Selection")

                selected_word_count = st.sidebar.slider(
                    "Number of Words to Include",
                    min_value=min_words,
                    max_value=max_words,
                    value=min(100, max_words),  # Default to 100 or max available
                    step=5,
                    help=f"Select between {min_words} and {max_words} words (60-100% of unique words)"
                )

                # Show percentage and word count info
                percentage = (selected_word_count / unique_words) * 100
                st.sidebar.info(f"**Selected**: {selected_word_count} words ({percentage:.1f}% of unique words)")

                # Show top words preview based on selection
                top_preview_words = dict(word_freq.most_common(min(20, selected_word_count)))

                with st.expander(f"📊 Preview of Top {min(20, selected_word_count)} Selected Words"):
                    for i, (word, freq) in enumerate(top_preview_words.items(), 1):
                        st.write(f"{i}. **{word}**: {freq} occurrences")
                    if selected_word_count > 20:
                        st.write(f"... and {selected_word_count - 20} more words")

                # Enhanced 3D Visual configuration
                st.sidebar.subheader("🎨 Visual Settings")

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

                # Font size information based on word count
                min_font, max_font = adjust_font_sizes_for_word_count(10, 40, selected_word_count)
                st.sidebar.info(f"**Auto Font Size Range**: {min_font}px - {max_font}px\n(Optimized for {selected_word_count} words)")

                # 3D Animation settings
                st.sidebar.subheader("🎬 Animation Settings")

                col3, col4 = st.sidebar.columns(2)
                with col3:
                    num_frames = st.slider("Rotation Frames", min_value=15, max_value=60, value=30)
                with col4:
                    duration = st.slider("Frame Speed (ms)", min_value=100, max_value=500, value=200)

                rotation_speed = st.sidebar.slider("Rotation Speed", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

                # Generate buttons
                col_gen1, col_gen2 = st.columns(2)

                with col_gen1:
                    if st.button(f"🖼️ Generate 3D WordCloud ({selected_word_count} words)"):
                        with st.spinner(f"Creating 3D word cloud with {selected_word_count} words..."):
                            static_wordcloud = create_static_3d_wordcloud(
                                text=processed_text,
                                selected_word_count=selected_word_count,
                                color_scheme=color_schemes[selected_color_scheme],
                                width=width,
                                height=height,
                                background_color=background_options[selected_background],
                                rotation_angle=np.pi/4,
                                sphere_scale=sphere_scale
                            )

                            if static_wordcloud:
                                st.subheader(f"🌍 3D WordCloud Sphere - {selected_word_count} Words ({percentage:.1f}%)")
                                st.image(static_wordcloud, caption=f"3D Word Cloud with {selected_word_count} words", use_column_width=True)

                                # Download button for static image
                                buf = BytesIO()
                                static_wordcloud.save(buf, format='PNG')
                                buf.seek(0)

                                st.download_button(
                                    label=f"💾 Download 3D Image ({selected_word_count} words)",
                                    data=buf.getvalue(),
                                    file_name=f"3d_wordcloud_{selected_word_count}words_{uploaded_file.name.replace('.pdf', '')}.png",
                                    mime="image/png"
                                )

                with col_gen2:
                    if st.button(f"🎬 Generate Rotating Sphere ({selected_word_count} words)"):
                        with st.spinner(f"Creating 3D rotating animation with {selected_word_count} words..."):
                            # Adjust frame count based on rotation speed
                            adjusted_frames = int(num_frames / rotation_speed)
                            adjusted_duration = int(duration * rotation_speed)

                            gif_buffer = create_3d_animated_wordcloud(
                                text=processed_text,
                                selected_word_count=selected_word_count,
                                color_scheme=color_schemes[selected_color_scheme],
                                num_frames=adjusted_frames,
                                duration=adjusted_duration,
                                width=width,
                                height=height,
                                background_color=background_options[selected_background],
                                sphere_scale=sphere_scale
                            )

                            if gif_buffer:
                                st.subheader(f"🌍 3D Rotating WordCloud Sphere - {selected_word_count} Words ({percentage:.1f}%)")
                                st.image(gif_buffer.getvalue(), caption=f"3D Rotating Word Cloud with {selected_word_count} words")

                                # Download button for GIF
                                st.download_button(
                                    label=f"💾 Download 3D Animated GIF ({selected_word_count} words)",
                                    data=gif_buffer.getvalue(),
                                    file_name=f"3d_rotating_wordcloud_{selected_word_count}words_{uploaded_file.name.replace('.pdf', '')}.gif",
                                    mime="image/gif"
                                )

                                st.success(f"🎉 3D rotating word cloud with {selected_word_count} words created successfully!")
                            else:
                                st.error("Failed to create 3D animated GIF")
            else:
                st.error("❌ No valid words found after preprocessing. Please check your document or stopword settings.")
        else:
            st.error("❌ Failed to extract text from PDF. Please check if the file is valid.")

    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload a PDF file to get started!")

        # Sample features showcase
        st.subheader("✨ Dynamic Word Count Features")

        feature_cols = st.columns(3)

        with feature_cols[0]:
            st.markdown("""
            **📈 Smart Word Selection**
            - 60-100% of unique words
            - Automatic range calculation
            - Performance-optimized limits
            - Real-time percentage display
            """)

        with feature_cols[1]:
            st.markdown("""
            **🎨 Adaptive Visualization**
            - Font sizes adjust to word count
            - Optimal readability maintained
            - Sphere density optimization
            - Professional quality output
            """)

        with feature_cols[2]:
            st.markdown("""
            **⚡ User Control**
            - Slider-based word selection
            - Live preview of selections
            - Word statistics display
            - Customizable file naming
            """)

        st.subheader("🎯 How Dynamic Word Count Works")
        st.markdown("""
        1. **Document Analysis**: Extract and analyze all unique words from your PDF
        2. **Range Calculation**: Determine 60-100% range based on available words
        3. **Smart Limits**: Apply performance caps (200-300 words max) for optimal rendering
        4. **Word Selection**: Use slider to choose exact number of words to include
        5. **Font Optimization**: Automatically adjust font sizes for selected word count
        6. **Sphere Generation**: Create Fibonacci distribution for selected words
        7. **Quality Output**: Generate high-quality 3D wordcloud with optimal density

        **Benefits:**
        - **More Detail**: Include up to 300 words vs. fixed 50
        - **Better Coverage**: Capture 60-100% of document vocabulary
        - **Flexible Control**: Adjust density based on document complexity
        - **Optimized Display**: Auto font sizing prevents overcrowding
        """)

if __name__ == "__main__":
    main()
