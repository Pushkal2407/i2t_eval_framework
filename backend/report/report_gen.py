class ReportGenerator:
    """
    Generates a structured report of the evaluation, including caption, similarity,
    L-infinity norm, and image quality metrics.
    """
    def generate_report(self, result: dict) -> str:
        report = (
            "=== Evaluation Report ===\n"
            f"Image Name: {result.get('image_name', 'N/A')}\n"
            f"Normal Caption: {result.get('normal_caption', 'N/A')}\n"
            f"Adversarial Caption: {result.get('adversarial_caption', 'N/A')}\n"
            f"Caption Similarity: {result.get('caption_similarity', 0):.4f}\n"
            f"L-infinity Norm: {result.get('l_inf_norm', 0):.6f}\n"
            f"MSE: {result.get('MSE', 'N/A')}\n"
            f"PSNR: {result.get('PSNR', 'N/A')}\n"
            f"SSIM: {result.get('SSIM', 'N/A')}\n"
        )
        return report
