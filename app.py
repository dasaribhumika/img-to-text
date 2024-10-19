import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io
import time

# Define custom CSS for black background and centered content
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    .centered-content {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 80vh;
    }
    </style>
    """, unsafe_allow_html=True
)

# Define a function to load the model (using caching to avoid reloading on every run)
@st.cache_resource
def load_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    return pipe

# Load the model
pipe = load_model()

# Streamlit app title
st.title("AI Prompt-A-Thon: Text-to-Image Generator")

# Input for the user prompt (allowing multiline input)
prompt = st.text_area(
    "Enter your prompt for the image generation (you can use multiline prompts):",
    """A futuristic urban hospital using AI-powered robots to assist in surgeries, 
    reducing errors, and providing personalized care to patients from all socioeconomic backgrounds, 
    while maintaining strict patient privacy and equitable access to healthcare."""
)

# Button to generate the image
if st.button("Generate Image"):
    # Container for centering the progress bar
    with st.container():
        st.markdown('<div class="centered-content">', unsafe_allow_html=True)
        
        # Initialize progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Simulate step-by-step progress
        for percent_complete in range(100):
            time.sleep(0.05)  # Artificial delay for progress bar visualization
            progress_bar.progress(percent_complete + 1)
            status_text.text(f"Generating image... {percent_complete + 1}%")

        # End of centered content div
        st.markdown('</div>', unsafe_allow_html=True)

    # Generate the image from the prompt
    with st.spinner("Finalizing image..."):
        image = pipe(prompt).images[0]

    # Display the generated image
    st.image(image, caption="Generated Image", use_column_width=True)

    # Create a download button for the generated image
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    # Download button
    st.download_button(
        label="Download Image",
        data=byte_im,
        file_name="generated_image.png",
        mime="image/png"
    )
