import os
import random
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from pycocotools.coco import COCO

# Import modular components
from models import BLIPModel
from eval import Evaluator
from attacks import PGDAttack, FGSMAttack
from report import ReportGenerator
from utils import display_results  # Import the display function
from data.coco_dataset.database_functions import display_random_coco_images


# Optional: Disable file watcher issues for torch (if needed)
os.environ["TORCH_DONT_LOAD_FALLBACK_MODULES"] = "1"

# Set Streamlit page configuration
st.set_page_config(
    page_title="Image-to-Text Evaluation",
    page_icon="🖼️",
    layout="wide"
)

# Inject custom CSS for a colorful, professional UI
st.markdown(
    """
    <style>
        .reportview-container {
            background: #f5f7fa;
        }
        .sidebar .sidebar-content {
            background: #dce8f1;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #1f77b4;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stButton > button {
            background-color: #1f77b4;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
        }
        .report-box {
            background: #ffffff;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #ccc;
            font-family: 'Segoe UI', sans-serif;
            color: #333;
        }
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
    return transform(image).unsqueeze(0).to(device, dtype=torch.float16)



def main():
    st.title("🖼️ Image-to-Text Evaluation Framework with Adversarial Attacks")
    
    # Sidebar: Dataset selection
    dataset_choice = st.sidebar.radio("Select Dataset", ["Custom Upload", "COCO 2017"])
    
    # Sidebar: Model and Attack configuration (common to both workflows)
    st.sidebar.header("🔧 Configuration")
    model_choice = st.sidebar.selectbox("📌 Select Model", ["BLIP-base", "BLIP-large"])
    attack_choice = st.sidebar.radio("Select Attack Method", ["PGD Attack", "FGSM Attack"])
    
    model_name = "Salesforce/blip-image-captioning-base" if model_choice == "BLIP-base" else "Salesforce/blip-image-captioning-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    st.sidebar.info("⏳ Loading model...")
    model = load_model(model_name, device)
    st.sidebar.success("✅ Model loaded!")
    
    if dataset_choice == "Custom Upload":
        # --- Existing Custom Upload Workflow ---
        st.subheader("📤 Upload an Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            image_tensor = preprocess_image(image, device)
            st.session_state.image_tensor = image_tensor
            st.session_state.image = image
            
            if st.button("📝 Generate Caption"):
                with st.spinner("⏳ Generating original caption..."):
                    evaluator = Evaluator(model)
                    original_result = evaluator.evaluate(image_tensor)
                    original_caption = original_result["caption"]
                    st.session_state.original_caption = original_caption
                    st.session_state.original_result = original_result
                st.success("✅ Caption Generated!")
                st.write("**📌 Original Caption:**", original_caption)
            
            if "original_caption" in st.session_state:
                st.subheader("⚡ Attack Parameters")
                if attack_choice == "PGD Attack":
                    epsilon = st.slider("🔧 Set Epsilon (Perturbation Limit)", 0.01, 0.2, 0.05, step=0.01)
                    alpha = st.slider("🔧 Set Alpha (Step Size)", 0.001, 0.05, 0.01, step=0.001)
                    num_steps = st.slider("🔄 Number of PGD Steps", 1, 20, 10)
                else:
                    epsilon = st.slider("🔧 Set Epsilon for FGSM", 0.01, 0.2, 0.05, step=0.01)
                
                if st.button("🚀 Run Attack"):
                    st.write("✅ Attack button clicked!")
                    with st.spinner("🔄 Running Attack..."):
                        try:
                            if attack_choice == "PGD Attack":
                                st.write("🚀 Initializing PGD Attack...")
                                attacker = PGDAttack(model, epsilon=epsilon, alpha=alpha, num_steps=num_steps)
                                adv_image_tensor = attacker.run(st.session_state.image_tensor, target_text="")
                            else:
                                st.write("🚀 Initializing FGSM Attack...")
                                attacker = FGSMAttack(model, epsilon=epsilon)
                                adv_image_tensor = attacker.run(st.session_state.image_tensor, target_text="")
                            
                            st.session_state.adv_image_tensor = adv_image_tensor
                            st.write("✅ Attack completed!")
                            
                            with st.spinner("⏳ Generating adversarial caption..."):
                                evaluator = Evaluator(model)
                                adversarial_result = evaluator.evaluate(adv_image_tensor)
                                adv_caption = adversarial_result["caption"]
                                st.session_state.adv_caption = adv_caption
                            st.write("✅ Adversarial caption generated:", adv_caption)
                            
                            l_inf_norm = (adv_image_tensor - st.session_state.image_tensor).abs().max().item()
                            st.session_state.l_inf_norm = l_inf_norm
                            st.write(f"📊 L-infinity norm of perturbation: {l_inf_norm}")
                            
                            reporter = ReportGenerator()
                            report = reporter.generate_report(st.session_state.original_result, adversarial_result, l_inf_norm)
                            st.session_state.report = report
                            
                            display_results(st.session_state.image, adv_image_tensor, st.session_state.original_caption, adv_caption)
                            
                            st.subheader("📜 Evaluation Report")
                            st.markdown(f"<div class='report-box'>{report}</div>", unsafe_allow_html=True)
                        
                        except Exception as e:
                            st.error(f"⚠️ Error running attack: {str(e)}")
                            import traceback
                            st.text(traceback.format_exc())
    
    else:  # COCO 2017 selected
        st.subheader("📂 COCO 2017 Evaluation")
        # Assume your COCO dataset is stored in the "data" folder
        coco_data_dir = st.text_input("COCO Data Directory", "data/coco_dataset")
        data_split = "val2017"
        
        # Button to show 5 random images from COCO 2017
        if st.button("🚀 Show 5 Random COCO Images"):
            with st.spinner("🔄 Loading random images..."):
                try:
                    display_random_coco_images(coco_data_dir, data_split, num_images=5, device=device)
                    st.success("✅ 5 Random COCO images loaded!")
                except Exception as e:
                    st.error(f"⚠️ Error loading random COCO images: {str(e)}")
        
        # Button to load one random COCO image for testing the attack
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
            if st.button("📝 Generate Caption"):
                with st.spinner("⏳ Generating original caption..."):
                    evaluator = Evaluator(model)
                    original_result = evaluator.evaluate(st.session_state.image_tensor)
                    original_caption = original_result["caption"]
                    st.session_state.original_caption = original_caption
                    st.session_state.original_result = original_result
                st.success("✅ Caption Generated!")
                st.write("**📌 Original Caption:**", original_caption)
            
            if "original_caption" in st.session_state:
                st.subheader("⚡ Attack Parameters")
                if attack_choice == "PGD Attack":
                    epsilon = st.slider("🔧 Set Epsilon (Perturbation Limit)", 0.01, 0.2, 0.05, step=0.01)
                    alpha = st.slider("🔧 Set Alpha (Step Size)", 0.001, 0.05, 0.01, step=0.001)
                    num_steps = st.slider("🔄 Number of PGD Steps", 1, 20, 10)
                else:
                    epsilon = st.slider("🔧 Set Epsilon for FGSM", 0.01, 0.2, 0.05, step=0.01)
                
                if st.button("🚀 Run Attack"):
                    st.write("✅ Attack button clicked!")
                    with st.spinner("🔄 Running Attack..."):
                        try:
                            if attack_choice == "PGD Attack":
                                st.write("🚀 Initializing PGD Attack...")
                                attacker = PGDAttack(model, epsilon=epsilon, alpha=alpha, num_steps=num_steps)
                                adv_image_tensor = attacker.run(st.session_state.image_tensor, target_text="")
                            else:
                                st.write("🚀 Initializing FGSM Attack...")
                                attacker = FGSMAttack(model, epsilon=epsilon)
                                adv_image_tensor = attacker.run(st.session_state.image_tensor, target_text="")
                            
                            st.session_state.adv_image_tensor = adv_image_tensor
                            st.write("✅ Attack completed!")
                            
                            with st.spinner("⏳ Generating adversarial caption..."):
                                evaluator = Evaluator(model)
                                adversarial_result = evaluator.evaluate(adv_image_tensor)
                                adv_caption = adversarial_result["caption"]
                                st.session_state.adv_caption = adv_caption
                            st.write("✅ Adversarial caption generated:", adv_caption)
                            
                            l_inf_norm = (adv_image_tensor - st.session_state.image_tensor).abs().max().item()
                            st.session_state.l_inf_norm = l_inf_norm
                            st.write(f"📊 L-infinity norm of perturbation: {l_inf_norm}")
                            
                            reporter = ReportGenerator()
                            report = reporter.generate_report(st.session_state.original_result, adversarial_result, l_inf_norm)
                            st.session_state.report = report
                            
                            display_results(st.session_state.image, adv_image_tensor, st.session_state.original_caption, adv_caption)
                            
                            st.subheader("📜 Evaluation Report")
                            st.markdown(f"<div class='report-box'>{report}</div>", unsafe_allow_html=True)
                        
                        except Exception as e:
                            st.error(f"⚠️ Error running attack: {str(e)}")
                            import traceback
                            st.text(traceback.format_exc())
        
if __name__ == "__main__":
    main()
