class ReportGenerator:
    """
    Generates a structured report of the evaluation.
    """
    def generate_report(self, original_result: dict, adversarial_result: dict, perturbation_norm: float) -> str:
        report = (
            f"=== Evaluation Report ===\n"
            f"Original Caption: {original_result.get('caption', 'N/A')}\n"
            f"Adversarial Caption: {adversarial_result.get('caption', 'N/A')}\n"
        )
        if 'target_text' in adversarial_result:
            report += f"Target Caption: {adversarial_result['target_text']}\n"
        report += f"L-infinity Norm of Perturbation: {perturbation_norm:.6f}\n"
        return report
