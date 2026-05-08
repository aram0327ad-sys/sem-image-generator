import os
import streamlit as st
from generator import run_generation

st.set_page_config(
    page_title="Synthetic SEM Image Generator",
    layout="centered"
)

st.title("Synthetic SEM Image Generator")

st.write(
    """
    This web app generates synthetic SEM images for polymer composite microstructure analysis.

    Enter the number of images you want to generate.
    The maximum number of images is **1000**.
    """
)

num_images = st.number_input(
    "Number of images to generate",
    min_value=1,
    max_value=1000,
    value=300,
    step=1,
    help="You can generate between 1 and 1000 images."
)

st.info("Input range: 1 to 1000 images")

st.subheader("Phase selection")

use_filler = st.checkbox("Filler 생성", value=True)
use_void = st.checkbox("Void 생성", value=True)

if not use_filler and not use_void:
    st.warning("Filler 또는 Void 중 최소 하나는 선택하는 것을 권장합니다.")


if st.button("Generate Dataset"):
    with st.spinner("Generating images... This may take several minutes depending on the number of images."):

        zip_path, output_dir = run_generation(
            num_images=num_images,
            use_filler=use_filler,
            use_void=use_void
        )

    st.success(f"Dataset generation completed: {num_images} images")

    with open(zip_path, "rb") as f:
        st.download_button(
            label="Download ZIP File",
            data=f,
            file_name=os.path.basename(zip_path),
            mime="application/zip"
        )

    st.write("Output folder:", output_dir)

