import os
import json
import csv
import torch
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torchvision import transforms
from PIL import Image
from difflib import SequenceMatcher
from tqdm import tqdm

def evaluate_perturbation(original_image, perturbed_image):
    """
    Compute pixel-level metrics (MSE, PSNR, SSIM) between the original and perturbed images.
    """
    # Ensure tensors are in shape [C, H, W]
    if original_image.dim() == 4:
        original_image = original_image.squeeze(0)
    if perturbed_image.dim() == 4:
        perturbed_image = perturbed_image.squeeze(0)
    
    # Resize the original image to match perturbed dimensions
    perturbed_height, perturbed_width = perturbed_image.shape[1:]
    original_image_resized = torch.nn.functional.interpolate(
        original_image.unsqueeze(0),
        size=(perturbed_height, perturbed_width),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)
    
    # Convert tensors to numpy arrays
    original_np = original_image_resized.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    perturbed_np = perturbed_image.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    
    mse = float(np.mean((original_np - perturbed_np) ** 2))
    psnr = float(compare_psnr(original_np, perturbed_np, data_range=1.0))
    
    try:
        ssim = float(compare_ssim(original_np, perturbed_np, data_range=1.0, channel_axis=2, win_size=7))
    except ValueError:
        min_dim = min(original_np.shape[0], original_np.shape[1])
        win_size = min(7, min_dim - (min_dim % 2) + 1)
        ssim = float(compare_ssim(original_np, perturbed_np, data_range=1.0, channel_axis=2, win_size=win_size))
    
    return {"MSE": mse, "PSNR": psnr, "SSIM": ssim}

def compute_caption_similarity(caption1, caption2):
    """
    Compute a simple similarity score between two captions.
    """
    return SequenceMatcher(None, caption1, caption2).ratio()

def load_image_as_tensor(image_path, device='cpu'):
    """
    Load an image from disk, resize, and convert it to a torch tensor.
    """
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor

def evaluate_single_image_from_tensor(image_tensor, model, attacker):
    """
    Given an image tensor, run the attack, generate captions, and compute metrics.
    Returns a dict with captions, similarity, perturbation metrics, and the attacked image.
    """
    # Generate normal caption
    normal_caption = model.generate_caption(image_tensor)
    
    # Run adversarial attack (attacker.run returns the perturbed image tensor)
    adv_image_tensor = attacker.run(image_tensor, target_text="")
    
    # Generate adversarial caption
    adv_caption = model.generate_caption(adv_image_tensor)
    
    # Compute image-level metrics
    metrics = evaluate_perturbation(image_tensor, adv_image_tensor)
    
    # Compute caption similarity
    caption_similarity = compute_caption_similarity(normal_caption, adv_caption)
    
    # Compute L-infinity norm of the perturbation
    l_inf_norm = (adv_image_tensor - image_tensor).abs().max().item()
    
    result = {
        "normal_caption": normal_caption,
        "adversarial_caption": adv_caption,
        "caption_similarity": caption_similarity,
        "MSE": metrics["MSE"],
        "PSNR": metrics["PSNR"],
        "SSIM": metrics["SSIM"],
        "l_inf_norm": l_inf_norm,
        "adv_image_tensor": adv_image_tensor
    }
    return result

def evaluate_single_image_from_path(image_path, model, attacker, device='cpu'):
    """
    Load an image from the given path and evaluate it.
    """
    image_tensor = load_image_as_tensor(image_path, device=device)
    result = evaluate_single_image_from_tensor(image_tensor, model, attacker)
    result["image_name"] = os.path.basename(image_path)
    return result

def evaluate_dataset(coco_dir, model, attacker, num_images=10, device='cpu'):
    """
    Evaluate a subset of the COCO dataset.
    Iterates over the specified number of images, computes metrics, and saves CSV/JSON reports.
    """
    ann_file = os.path.join(coco_dir, "annotations", "captions_val2017.json")
    image_dir = os.path.join(coco_dir, "val2017")
    from pycocotools.coco import COCO
    coco_caps = COCO(ann_file)
    valid_imgIds = coco_caps.getImgIds()[:num_images]
    
    all_results = []
    
    for img_id in tqdm(valid_imgIds, desc="Evaluating Dataset"):
        img_info = coco_caps.loadImgs(img_id)[0]
        image_path = os.path.join(image_dir, img_info["file_name"])
        if not os.path.exists(image_path):
            continue
        result = evaluate_single_image_from_path(image_path, model, attacker, device=device)
        result["image_id"] = img_id
        all_results.append(result)
    
    # Remove non-serializable tensor data from each result.
    for res in all_results:
        if "adv_image_tensor" in res:
            del res["adv_image_tensor"]

    # Save results to CSV.
    csv_file = os.path.join(coco_dir, "evaluation_results.csv")
    fieldnames = [
        "image_id", "image_name", "normal_caption", "adversarial_caption", "caption_similarity",
        "MSE", "PSNR", "SSIM", "l_inf_norm"
    ]
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        filtered_results = [{k: row[k] for k in fieldnames if k in row} for row in all_results]
        writer.writerows(filtered_results)
    
    # Save results to JSON.
    json_file = os.path.join(coco_dir, "evaluation_results.json")
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    return all_results, csv_file, json_file
