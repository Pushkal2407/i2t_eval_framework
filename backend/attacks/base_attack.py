import torch

class BaseAttack:
    """Interface for adversarial attacks."""
    def run(self, image: torch.Tensor, target_text: str, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement run()")
