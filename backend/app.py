import os
import random
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from pycocotools.coco import COCO
import pandas as pd
from models import BLIPModel
from eval import Evaluator  # (if needed for legacy evaluation)
from eval import evaluate_single_image_from_tensor, evaluate_dataset
from attacks import PGDAttack, FGSMAttack
from report import ReportGenerator
from utils import display_results  # Import the display function
from data.coco_dataset.database_functions import display_random_coco_images

st.set_page_config(
    page_title="Image-to-Text Evaluation",
    page_icon="🖼️",
    layout="wide"
)

st.markdown(
    """
    <style>
        .reportview-container { background: #f5f7fa; }
        .sidebar .sidebar-content { background: #dce8f1; }
        h1, h2, h3, h4, h5, h6 { color: #1f77b4; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .stButton > button { background-color: #1f77b4; color: white; border-radius: 10px; border: none; padding: 10px 20px; font-size: 16px; font-weight: bold; }
        .report-box { background: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #ccc; font-family: 'Segoe UI', sans-serif; color: #333; }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_model(model_name: str, device: str):
    return BLIPModel(model_name=model_name, device=device)

def preprocess_image(image, device):
    """Preprocess image into a tensor."""
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device, dtype=torch.float32)

def main():
    st.title("🖼️ Image-to-Text Evaluation Framework with Adversarial Attacks")
    
    # Sidebar: Choose evaluation mode
    evaluation_mode = st.sidebar.radio("Select Evaluation Mode", ["Single Image Evaluation", "Dataset Evaluation"])
    
    # For single image mode, allow a choice between custom upload and COCO 2017 images
    if evaluation_mode == "Single Image Evaluation":
        dataset_choice = st.sidebar.radio("Select Dataset", ["Custom Upload", "COCO 2017"])
    else:
        dataset_choice = "COCO 2017"
    
    # Sidebar: Model and Attack configuration (common)
    st.sidebar.header("🔧 Configuration")
    model_choice = st.sidebar.selectbox("📌 Select Model", ["BLIP-base", "BLIP-large"])
    attack_choice = st.sidebar.radio("Select Attack Method", ["PGD Attack", "FGSM Attack"])
    
    model_name = "Salesforce/blip-image-captioning-base" if model_choice == "BLIP-base" else "Salesforce/blip-image-captioning-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    st.sidebar.info("⏳ Loading model...")
    model = load_model(model_name, device)
    st.sidebar.success("✅ Model loaded!")
    
    # Configure attacker based on the chosen attack method
    if attack_choice == "PGD Attack":
        epsilon = st.sidebar.slider("🔧 Set Epsilon (Perturbation Limit)", 0.01, 0.2, 0.05, step=0.01)
        alpha = st.sidebar.slider("🔧 Set Alpha (Step Size)", 0.001, 0.05, 0.01, step=0.001)
        num_steps = st.sidebar.slider("🔄 Number of PGD Steps", 1, 20, 10)
        attacker = PGDAttack(model, epsilon=epsilon, alpha=alpha, num_steps=num_steps)
    else:
        epsilon = st.sidebar.slider("🔧 Set Epsilon for FGSM", 0.01, 0.2, 0.05, step=0.01)
        attacker = FGSMAttack(model, epsilon=epsilon)
    
    if dataset_choice == "Custom Upload":
        st.subheader("📤 Upload an Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            image_tensor = preprocess_image(image, device)
            st.session_state.image_tensor = image_tensor
            st.session_state.image = image
            
            if st.button("📝 Run Single Image Evaluation"):
                with st.spinner("⏳ Evaluating image..."):
                    result = evaluate_single_image_from_tensor(st.session_state.image_tensor, model, attacker)
                    st.session_state.eval_result = result
                    reporter = ReportGenerator()
                    report = reporter.generate_report(result)
                    st.session_state.report = report
                st.success("✅ Evaluation Complete!")
                st.markdown(f"<div class='report-box'><pre>{report}</pre></div>", unsafe_allow_html=True)
                display_results(st.session_state.image, result["adv_image_tensor"], result["normal_caption"], result["adversarial_caption"])
                st.download_button("Download Report", report, file_name="evaluation_report.txt")
    
    else:  # COCO 2017 workflow for both single image and dataset evaluation
        st.subheader("📂 COCO 2017 Evaluation")
        coco_data_dir = st.text_input("COCO Data Directory", "data/coco_dataset")
        
        # Option to show a few random COCO images
        if st.button("🚀 Show 5 Random COCO Images"):
            with st.spinner("🔄 Loading random images..."):
                try:
                    display_random_coco_images(coco_data_dir, "val2017", num_images=5, device=device)
                    st.success("✅ 5 Random COCO images loaded!")
                except Exception as e:
                    st.error(f"⚠️ Error loading random COCO images: {str(e)}")
        
        if evaluation_mode == "Single Image Evaluation":
            if st.button("🚀 Load a Random COCO Image for Attack"):
                with st.spinner("🔄 Loading a random COCO image..."):
                    try:
                        ann_file = os.path.join(coco_data_dir, "annotations", "captions_val2017.json")
                        coco_caps = COCO(ann_file)
                        img_ids = coco_caps.getImgIds()
                        selected_img_id = random.choice(img_ids)
                        img_info = coco_caps.loadImgs(selected_img_id)[0]
                        image_path = os.path.join(coco_data_dir, "val2017", img_info["file_name"])
                        image = Image.open(image_path).convert("RGB")
                        st.session_state.image = image
                        image_tensor = preprocess_image(image, device)
                        st.session_state.image_tensor = image_tensor
                        st.success(f"✅ Loaded COCO Image: {img_info['file_name']}")
                        st.image(image, caption=f"COCO Image: {img_info['file_name']}", use_container_width=True)
                    except Exception as e:
                        st.error(f"⚠️ Error loading COCO image: {str(e)}")
            
            if "image_tensor" in st.session_state:
                if st.button("📝 Run Single Image Evaluation"):
                    with st.spinner("⏳ Evaluating image..."):
                        result = evaluate_single_image_from_tensor(st.session_state.image_tensor, model, attacker)
                        st.session_state.eval_result = result
                        reporter = ReportGenerator()
                        report = reporter.generate_report(result)
                        st.session_state.report = report
                    st.success("✅ Evaluation Complete!")
                    st.markdown(f"<div class='report-box'><pre>{report}</pre></div>", unsafe_allow_html=True)
                    display_results(st.session_state.image, result["adv_image_tensor"], result["normal_caption"], result["adversarial_caption"])
                    st.download_button("Download Report", report, file_name="evaluation_report.txt")
        
        else:  # Dataset Evaluation mode
            num_images_dataset = st.number_input("Number of images for dataset evaluation", min_value=1, max_value=1000, value=10, step=1)
            if st.button("🚀 Run Full Dataset Evaluation"):
                with st.spinner("⏳ Evaluating dataset..."):
                    results, csv_path, json_path = evaluate_dataset(coco_data_dir, model, attacker, num_images=int(num_images_dataset), device=device)
                st.success("✅ Dataset Evaluation Complete!")
                
                # Display a summary table using pandas
                df = pd.DataFrame(results)
                st.dataframe(df)
                
                # Provide download buttons for CSV and JSON reports
                with open(csv_path, 'rb') as f:
                    csv_data = f.read()
                st.download_button("Download CSV Report", csv_data, file_name="evaluation_results.csv")
                with open(json_path, 'rb') as f:
                    json_data = f.read()
                st.download_button("Download JSON Report", json_data, file_name="evaluation_results.json")
    
if __name__ == "__main__":
    main()
