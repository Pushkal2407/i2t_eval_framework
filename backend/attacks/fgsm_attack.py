import torch
from .base_attack import BaseAttack  # Adjust this import if your BaseAttack is defined elsewhere

class FGSMAttack(BaseAttack):
    """
    Implements an FGSM attack for BLIP.
    
    Args:
        model: Instance of BLIPModel.
        epsilon: Perturbation magnitude for the FGSM attack.
    """
    def __init__(self, model, epsilon: float = 0.05):
        self.model = model
        self.epsilon = epsilon

    def run(self, image: torch.Tensor, target_text: str) -> torch.Tensor:
        """
        Applies the FGSM attack on the given image tensor.
        
        Args:
            image (torch.Tensor): Input image tensor.
            target_text (str): Target text for teacher forcing (can be an empty string for untargeted attacks).
            
        Returns:
            torch.Tensor: The perturbed image tensor.
        """
        # Clone the original image and enable gradients
        original_image = image.clone().detach()
        adv_image = original_image.clone().requires_grad_(True)
        
        # Process target text using the model's processor
        text_inputs = self.model.processor(
            text=[target_text],
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True
        )
        text_inputs = {k: v.to(self.model.device) for k, v in text_inputs.items()}
        
        # Prepare inputs for the model (using teacher forcing with "labels")
        inputs = {
            "pixel_values": adv_image,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "labels": text_inputs["input_ids"],
        }
        
        # Forward pass and compute loss
        outputs = self.model.model(**inputs)
        loss = outputs.loss
        print(f"FGSM Attack, Loss: {loss.item():.4f}")
        
        self.model.model.zero_grad()
        loss.backward()
        
        # Perform FGSM update: single step using the sign of the gradient
        with torch.no_grad():
            grad_sign = adv_image.grad.sign()
            perturbed = adv_image - self.epsilon * grad_sign
            # Project the perturbation onto the epsilon ball relative to the original image
            delta = torch.clamp(perturbed - original_image, -self.epsilon, self.epsilon)
            perturbed = original_image + delta
            perturbed = torch.clamp(perturbed, 0, 1)
        perturbed = perturbed.detach()
        return perturbed

    def generate_adversarial_caption_and_image(self, image: torch.Tensor, target_text: str) -> (str, torch.Tensor):
        """
        Generates an adversarial caption for the given image by applying the FGSM attack
        and then using the model to decode the caption from the perturbed image.
        
        Args:
            image (torch.Tensor): Preprocessed input image tensor.
            target_text (str): Target text (for teacher forcing; can be empty for untargeted attack).
            
        Returns:
            tuple: (adversarial_caption (str), perturbed_image_tensor (torch.Tensor))
        """
        # Run the FGSM attack to get the perturbed image tensor
        perturbed_image = self.run(image, target_text)
        
        # Generate adversarial caption using the model's generate method
        with torch.no_grad():
            adversarial_inputs = self.model.processor(
                return_tensors="pt",
                text=""
            ).to(self.model.device)
            adversarial_inputs["pixel_values"] = perturbed_image
            caption_ids = self.model.model.generate(**adversarial_inputs)
            adversarial_caption = self.model.processor.decode(caption_ids[0], skip_special_tokens=True)
        
        return adversarial_caption, perturbed_image