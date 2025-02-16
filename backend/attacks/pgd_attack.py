import torch
from .base_attack import BaseAttack  # Adjust this import if your BaseAttack is defined elsewhere

class PGDAttack(BaseAttack):
    """
    Implements a targeted PGD attack for BLIP.
    
    Args:
        model: Instance of BLIPModel.
        epsilon: Maximum L-infinity perturbation.
        alpha: Step size.
        num_steps: Number of PGD iterations.
    """
    def __init__(self, model, epsilon: float = 0.05, alpha: float = 0.01, num_steps: int = 5):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps

    def run(self, image: torch.Tensor, target_text: str) -> torch.Tensor:
        adv_image = image.clone().detach()
        
        # Process target text once using the model's processor
        text_inputs = self.model.processor(
            text=[target_text],
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True
        )
        text_inputs = {k: v.to(self.model.device) for k, v in text_inputs.items()}
        
        for step in range(self.num_steps):
            adv_image.requires_grad_(True)
            
            # Prepare inputs for the model (using teacher forcing with "labels")
            inputs = {
                "pixel_values": adv_image,
                "input_ids": text_inputs["input_ids"],
                "attention_mask": text_inputs["attention_mask"],
                "labels": text_inputs["input_ids"],
            }
            outputs = self.model.model(**inputs)
            loss = outputs.loss

            print(f"Step {step+1}/{self.num_steps}, Loss: {loss.item():.4f}")

            self.model.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                grad_sign = adv_image.grad.sign()
                adv_image = adv_image - self.alpha * grad_sign
                # Project onto epsilon ball
                delta = torch.clamp(adv_image - image, -self.epsilon, self.epsilon)
                adv_image = image + delta
                adv_image = torch.clamp(adv_image, 0, 1)
            adv_image = adv_image.detach()

        return adv_image
