import os
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# Import modular components
from models import BLIPModel
from eval import Evaluator
from attacks import PGDAttack, FGSMAttack
from report import ReportGenerator
from utils import display_results  # Import the display function

# Optional: Disable file watcher issues for torch (if needed)
os.environ["TORCH_DONT_LOAD_FALLBACK_MODULES"] = "1"

# Set Streamlit page configuration
st.set_page_config(
    page_title="Image-to-Text Evaluation",
    page_icon="🖼️",
    layout="wide"
)

# Inject custom CSS for a colorful and professional UI
st.markdown(
    """
    <style>
        /* Overall background */
        .reportview-container {
            background: #f5f7fa;
        }
        .sidebar .sidebar-content {
            background: #dce8f1;
        }
        /* Headings and text */
        h1, h2, h3, h4, h5, h6 {
            color: #1f77b4;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .css-1d391kg {  /* container style for inputs/buttons */
            font-family: 'Segoe UI', sans-serif;
        }
        /* Buttons */
        .stButton > button {
            background-color: #1f77b4;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
        }
        .stSlider > div {
            font-family: 'Segoe UI', sans-serif;
        }
        /* Report styling */
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
    
    # Sidebar: Model selection and Attack method selection
    st.sidebar.header("🔧 Configuration")
    model_choice = st.sidebar.selectbox("📌 Select Model", ["BLIP-base", "BLIP-large"])
    attack_choice = st.sidebar.radio("Select Attack Method", ["PGD Attack", "FGSM Attack"])
    
    model_name = "Salesforce/blip-image-captioning-base" if model_choice == "BLIP-base" else "Salesforce/blip-image-captioning-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    st.sidebar.info("⏳ Loading model...")
    model = load_model(model_name, device)
    st.sidebar.success("✅ Model loaded!")
    
    # Image Upload
    st.subheader("📤 Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Preprocess image and store it in session state
        image_tensor = preprocess_image(image, device)
        st.session_state.image_tensor = image_tensor
        st.session_state.image = image  # Save original image for later display
        
        # Section: Generate Original Caption
        if st.button("📝 Generate Caption"):
            with st.spinner("⏳ Generating original caption..."):
                evaluator = Evaluator(model)
                original_result = evaluator.evaluate(image_tensor)
                original_caption = original_result["caption"]
                st.session_state.original_caption = original_caption
                st.session_state.original_result = original_result
            st.success("✅ Caption Generated!")
            st.write("**📌 Original Caption:**", original_caption)
        
        # Show attack options only if caption has been generated
        if "original_caption" in st.session_state:
            st.subheader("⚡ Attack Parameters")
            # Attack method-specific parameters
            if attack_choice == "PGD Attack":
                epsilon = st.slider("🔧 Set Epsilon (Perturbation Limit)", 0.01, 0.2, 0.05, step=0.01)
                alpha = st.slider("🔧 Set Alpha (Step Size)", 0.001, 0.05, 0.01, step=0.001)
                num_steps = st.slider("🔄 Number of PGD Steps", 1, 20, 10)
            else:  # FGSM Attack
                epsilon = st.slider("🔧 Set Epsilon for FGSM", 0.01, 0.2, 0.05, step=0.01)
            
            # Attack button
            if st.button("🚀 Run Attack"):
                st.write("✅ Attack button clicked!")  # Debug print
                with st.spinner("🔄 Running Attack..."):
                    try:
                        # Ensure necessary state is available
                        if "image_tensor" not in st.session_state:
                            st.error("Image not processed. Please upload an image.")
                            return
                        if "original_caption" not in st.session_state:
                            st.error("Original caption not generated. Please generate caption first.")
                            return
                        
                        # Execute the selected attack
                        if attack_choice == "PGD Attack":
                            st.write("🚀 Initializing PGD Attack...")
                            attacker = PGDAttack(model, epsilon=epsilon, alpha=alpha, num_steps=num_steps)
                            adv_image_tensor = attacker.run(st.session_state.image_tensor, target_text="")  # Untargeted attack
                        else:  # FGSM Attack
                            st.write("🚀 Initializing FGSM Attack...")
                            attacker = FGSMAttack(model, epsilon=epsilon)
                            adv_image_tensor = attacker.run(st.session_state.image_tensor, target_text="")  # Untargeted attack
                        
                        st.session_state.adv_image_tensor = adv_image_tensor
                        st.write("✅ Attack completed!")
                        
                        # Generate caption for adversarial image
                        with st.spinner("⏳ Generating adversarial caption..."):
                            evaluator = Evaluator(model)  # Re-create evaluator
                            adversarial_result = evaluator.evaluate(adv_image_tensor)
                            adv_caption = adversarial_result["caption"]
                            st.session_state.adv_caption = adv_caption
                        st.write("✅ Adversarial caption generated:", adv_caption)
                        
                        # Compute perturbation norm
                        l_inf_norm = (adv_image_tensor - st.session_state.image_tensor).abs().max().item()
                        st.session_state.l_inf_norm = l_inf_norm
                        st.write(f"📊 L-infinity norm of perturbation: {l_inf_norm}")
                        
                        # Generate report
                        reporter = ReportGenerator()
                        report = reporter.generate_report(st.session_state.original_result, adversarial_result, l_inf_norm)
                        st.session_state.report = report
                        
                        # Display results with a styled container
                        display_results(st.session_state.image, adv_image_tensor, st.session_state.original_caption, adv_caption)
                        
                        st.subheader("📜 Evaluation Report")
                        st.markdown(f"<div class='report-box'>{report}</div>", unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"⚠️ Error running attack: {str(e)}")
                        import traceback
                        st.text(traceback.format_exc())
        
if __name__ == "__main__":
    main()
