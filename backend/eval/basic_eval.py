class Evaluator:
    """
    Evaluates a given model by generating captions.
    """
    def __init__(self, model):
        self.model = model

    def evaluate(self, image, target_text: str = None) -> dict:
        caption = self.model.generate_caption(image)
        result = {"caption": caption}
        if target_text is not None:
            result["target_text"] = target_text
        return result
