import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

class BaseImageToTextModel:
    """Interface for image-to-text models."""
    def generate_caption(self, image: torch.Tensor, **kwargs) -> str:
        raise NotImplementedError("Subclasses must implement generate_caption()")


class BLIPModel(BaseImageToTextModel):
    """BLIP implementation of the image-to-text model."""
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", device: str = "cuda", use_fp16: bool = True):
        self.device = device
        dtype = torch.float16 if use_fp16 else torch.float32
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype
        ).to(self.device).eval()

    def generate_caption(self, image: torch.Tensor, max_length: int = 30, num_beams: int = 5) -> str:
        generated_ids = self.model.generate(
            pixel_values=image,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return caption
