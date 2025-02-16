import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

def display_results(original_image, adv_image_tensor, original_caption, adv_caption):
    """Displays both images (original and adversarial) with their captions in Streamlit."""
    st.subheader("📷 Comparison: Original vs Adversarial Image")
    col1, col2 = st.columns(2)

    with col1:
        st.image(original_image, caption=f"Original Image\nCaption: {original_caption}", use_container_width=True)

    with col2:
        adv_img_pil = transforms.ToPILImage()(adv_image_tensor.squeeze().cpu())
        st.image(adv_img_pil, caption=f"Adversarial Image\nCaption: {adv_caption}", use_container_width=True)
