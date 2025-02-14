import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Import modular components
from models import BLIPModel
from eval import Evaluator
from attacks import PGDAttack
from report import ReportGenerator

@st.cache_resource
def load_model(model_name: str, device: str):
    return BLIPModel(model_name=model_name, device=device)

def preprocess_image(image, device):
    """Preprocess image into a tensor"""
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device, dtype=torch.float16)

def display_results(original_image, adv_image_tensor, original_caption, adv_caption):
    """Displays both images and captions side by side"""
    st.subheader("📷 Comparison: Original vs Adversarial Image")
    col1, col2 = st.columns(2)

    with col1:
        st.image(original_image, caption=f"Original Image\nCaption: {original_caption}", use_column_width=True)

    with col2:
        adv_img_pil = transforms.ToPILImage()(adv_image_tensor.squeeze().cpu())
        st.image(adv_img_pil, caption=f"Adversarial Image\nCaption: {adv_caption}", use_column_width=True)

def main():
    st.title("🖼️ Image-to-Text Evaluation Framework with Adversarial Attacks")
    
    # Sidebar: Model selection
    st.sidebar.header("🔧 Configuration")
    model_choice = st.sidebar.selectbox("📌 Select Model", ["BLIP-base", "BLIP-large"])
    
    model_name = "Salesforce/blip-image-captioning-base" if model_choice == "BLIP-base" else "Salesforce/blip-image-captioning-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    st.sidebar.info("⏳ Loading model...")
    model = load_model(model_name, device)
    st.sidebar.success("✅ Model loaded!")

    # Image Upload
    st.subheader("📤 Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        image_tensor = preprocess_image(image, device)
        
        # Step 1: Generate Original Caption
        if st.button("📝 Generate Caption"):
            with st.spinner("⏳ Generating original caption..."):
                evaluator = Evaluator(model)
                original_result = evaluator.evaluate(image_tensor)
                original_caption = original_result["caption"]
            
            st.success("✅ Caption Generated!")
            st.write("**📌 Original Caption:**", original_caption)

            # Step 2: Perform PGD Attack
            st.subheader("⚡ Perform Adversarial Attack")
            epsilon = st.slider("🔧 Set Epsilon (Perturbation Limit)", 0.01, 0.2, 0.05, 0.01)
            alpha = st.slider("🔧 Set Alpha (Step Size)", 0.001, 0.05, 0.01, 0.001)
            num_steps = st.slider("🔄 Number of PGD Steps", 1, 20, 10)

            if st.button("🚀 Run PGD Attack"):
                with st.spinner("🔄 Running PGD Attack..."):
                    attacker = PGDAttack(model, epsilon=epsilon, alpha=alpha, num_steps=num_steps)
                    adv_image_tensor = attacker.run(image_tensor, target_text="")  # Blank target for untargeted attack
                
                st.success("✅ Attack Completed!")

                # Step 3: Generate Caption for Adversarial Image
                with st.spinner("⏳ Generating adversarial caption..."):
                    adversarial_result = evaluator.evaluate(adv_image_tensor)
                    adv_caption = adversarial_result["caption"]

                # Compute perturbation norm
                l_inf_norm = (adv_image_tensor - image_tensor).abs().max().item()

                # Step 4: Display Results
                display_results(image, adv_image_tensor, original_caption, adv_caption)
                
                # Step 5: Generate Report
                # st.subheader("📜 Evaluation Report")
                # report = ReportGenerator().generate_report(original_result, adversarial_result, l_inf_norm)
                # st.text(report)

if __name__ == "__main__":
    main()
