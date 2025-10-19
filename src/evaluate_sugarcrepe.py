"""
SugarCREPE Evaluation for CLIP and CoN-CLIP
Evaluates compositional understanding via image-text retrieval on:
- Replace: Object, Attribute, Relation
- Add: Object, Attribute
- Swap: Object, Attribute
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import clip
from tqdm import tqdm
import json
import os
from PIL import Image


class SugarCREPEDataset(Dataset):
    """
    Dataset for SugarCREPE evaluation.
    Each sample contains an image, true caption, and a negative (hard) caption.
    """
    def __init__(self, data_root, split, transform=None):
        """
        Args:
            data_root: Root directory containing SugarCREPE data
            split: One of ['replace_obj', 'replace_att', 'replace_rel', 
                          'add_obj', 'add_att', 'swap_obj', 'swap_att']
            transform: Image preprocessing transform
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        
        # Load annotations - check both data/ subdirectory and root
        anno_path = os.path.join(data_root, "data", f"{split}.json")
        if not os.path.exists(anno_path):
            anno_path = os.path.join(data_root, f"{split}.json")
        
        with open(anno_path, 'r') as f:
            data = json.load(f)
        
        # Handle both dict and list formats
        if isinstance(data, dict):
            # If it's a dict, convert to list of samples
            self.annotations = []
            for key, value in data.items():
                if isinstance(value, dict):
                    # Each value should have 'filename', 'caption', 'negative_caption'
                    self.annotations.append(value)
                else:
                    # If structure is different, try to parse it
                    self.annotations.append({
                        'filename': value.get('filename', value.get('image_file', '')),
                        'caption': value.get('caption', value.get('true_caption', '')),
                        'negative_caption': value.get('negative_caption', value.get('false_caption', ''))
                    })
        elif isinstance(data, list):
            self.annotations = data
        else:
            raise ValueError(f"Unexpected JSON format for {split}")
        
        print(f"Loaded {len(self.annotations)} samples for {split}")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        anno = self.annotations[idx]
        
        # Handle different possible key names
        img_filename = anno.get('filename', anno.get('image_file', anno.get('image', '')))
        
        # Load image - handle both absolute and relative paths
        if img_filename.startswith('images/'):
            img_path = os.path.join(self.data_root, img_filename)
        else:
            img_path = os.path.join(self.data_root, 'images', img_filename)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # Try alternative path
            img_path = os.path.join(self.data_root, img_filename)
            image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        # Get captions - handle different possible key names
        true_caption = anno.get('caption', anno.get('true_caption', ''))
        negative_caption = anno.get('negative_caption', anno.get('false_caption', ''))
        
        return image, true_caption, negative_caption


def collate_fn(batch):
    """Collate function for DataLoader"""
    images = torch.stack([item[0] for item in batch], dim=0)
    true_captions = [item[1] for item in batch]
    negative_captions = [item[2] for item in batch]
    return images, true_captions, negative_captions


@torch.no_grad()
def evaluate_split(model, loader, device):
    """
    Evaluate on a single SugarCREPE split.
    
    For each image, compute similarity with true caption and negative caption.
    R@1 = percentage of times true caption has higher similarity.
    
    Args:
        model: CLIP model
        loader: DataLoader for the split
        device: torch device
    
    Returns:
        r_at_1: Retrieval accuracy (R@1)
    """
    model.eval()
    correct = 0
    total = 0
    
    for images, true_caps, neg_caps in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        
        # Tokenize captions
        true_tokens = clip.tokenize(true_caps, truncate=True).to(device)
        neg_tokens = clip.tokenize(neg_caps, truncate=True).to(device)
        
        # Encode images and text
        image_features = model.encode_image(images)
        true_text_features = model.encode_text(true_tokens)
        neg_text_features = model.encode_text(neg_tokens)
        
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        true_text_features = F.normalize(true_text_features, dim=-1)
        neg_text_features = F.normalize(neg_text_features, dim=-1)
        
        # Compute similarities (diagonal elements only for batch)
        sim_true = (image_features * true_text_features).sum(dim=-1)
        sim_neg = (image_features * neg_text_features).sum(dim=-1)
        
        # R@1: true caption should have higher similarity
        correct += (sim_true > sim_neg).sum().item()
        total += images.size(0)
    
    r_at_1 = 100.0 * correct / total
    return r_at_1


def evaluate_all_splits(model, preprocess, data_root, device, batch_size=32, num_workers=4):
    """
    Evaluate model on all SugarCREPE splits.
    
    Returns:
        results: Dict containing R@1 for each split
    """
    splits = [
        'replace_obj', 'replace_att', 'replace_rel',
        'add_obj', 'add_att',
        'swap_obj', 'swap_att'
    ]
    
    split_names = {
        'replace_obj': 'Replace Object',
        'replace_att': 'Replace Attribute',
        'replace_rel': 'Replace Relation',
        'add_obj': 'Add Object',
        'add_att': 'Add Attribute',
        'swap_obj': 'Swap Object',
        'swap_att': 'Swap Attribute'
    }
    
    results = {}
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Evaluating: {split_names[split]}")
        print(f"{'='*60}")
        
        try:
            # Load dataset
            dataset = SugarCREPEDataset(data_root, split, transform=preprocess)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_fn
            )
            
            # Evaluate
            r_at_1 = evaluate_split(model, loader, device)
            results[split] = r_at_1
            
            print(f"{split_names[split]}: {r_at_1:.2f}%")
            
        except Exception as e:
            print(f"Error evaluating {split}: {e}")
            results[split] = None
    
    return results


def print_results_table(all_results):
    """Print results in paper format (Table 5)"""
    print("\n" + "="*100)
    print("SugarCREPE Evaluation Results (R@1)")
    print("="*100)
    
    # Column headers
    print(f"{'Model':<20} | {'Replace':<45} | {'Add':<25} | {'Swap':<25}")
    print(f"{'':<20} | {'Object':>12} {'Attribute':>12} {'Relation':>12} | {'Object':>12} {'Attribute':>12} | {'Object':>12} {'Attribute':>12}")
    print("-"*100)
    
    # Print results for each model
    for model_name, results in all_results.items():
        row = f"{model_name:<20} |"
        
        # Replace
        row += f" {results.get('replace_obj', 0.0):>11.2f}"
        row += f" {results.get('replace_att', 0.0):>12.2f}"
        row += f" {results.get('replace_rel', 0.0):>12.2f} |"
        
        # Add
        row += f" {results.get('add_obj', 0.0):>11.2f}"
        row += f" {results.get('add_att', 0.0):>12.2f} |"
        
        # Swap
        row += f" {results.get('swap_obj', 0.0):>11.2f}"
        row += f" {results.get('swap_att', 0.0):>12.2f}"
        
        print(row)
    
    print("="*100)
    
    # Calculate and print averages per category
    print("\nCategory Averages:")
    for model_name, results in all_results.items():
        replace_avg = sum([results.get(k, 0) for k in ['replace_obj', 'replace_att', 'replace_rel']]) / 3
        add_avg = sum([results.get(k, 0) for k in ['add_obj', 'add_att']]) / 2
        swap_avg = sum([results.get(k, 0) for k in ['swap_obj', 'swap_att']]) / 2
        overall_avg = sum([v for v in results.values() if v is not None]) / len([v for v in results.values() if v is not None])
        
        print(f"{model_name:<20}: Replace={replace_avg:.2f}%, Add={add_avg:.2f}%, Swap={swap_avg:.2f}%, Overall={overall_avg:.2f}%")


def load_model(model_name, checkpoint_path=None, device="cuda"):
    """Load CLIP or CoN-CLIP model"""
    # Extract actual CLIP model name
    if model_name.startswith("CLIP-"):
        clip_name = model_name.replace("CLIP-", "")
    elif model_name.startswith("CoN-CLIP-"):
        clip_name = model_name.replace("CoN-CLIP-", "")
    else:
        clip_name = model_name
    
    print(f"Loading CLIP base model: {clip_name}")
    model, preprocess = clip.load(clip_name, device=device)
    
    # Load CoN-CLIP checkpoint if provided
    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = model.float()
        
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
    
    model = model.to(device)
    model.eval()
    return model, preprocess


def main(args):
    device = args.device
    
    # Models to evaluate
    models_config = {}
    
    if args.model:
        # Single model evaluation
        models_config[args.model] = args.ckpt
    else:
        # Compare multiple models
        models_config = {
            "CLIP-ViT-B/32": None,
            "CoN-CLIP-ViT-B/32": args.conclip_b32_ckpt,
            "CLIP-ViT-B/16": None,
            "CoN-CLIP-ViT-B/16": args.conclip_b16_ckpt,
            "CLIP-ViT-L/14": None,
            "CoN-CLIP-ViT-L/14": args.conclip_l14_ckpt,
        }
    
    all_results = {}
    
    for model_name, checkpoint_path in models_config.items():
        print(f"\n{'#'*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'#'*60}")
        
        # Load model
        model, preprocess = load_model(model_name, checkpoint_path, device)
        
        # Evaluate on all splits
        results = evaluate_all_splits(
            model, preprocess, args.data_root, device,
            batch_size=args.batch_size, num_workers=args.num_workers
        )
        
        all_results[model_name] = results
    
    # Print results table
    print_results_table(all_results)
    
    # Save results
    if args.output:
        torch.save(all_results, args.output)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CLIP/CoN-CLIP on SugarCREPE")
    
    # Data args
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to SugarCREPE dataset root")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Model args
    parser.add_argument("--model", type=str, default=None,
                        help="Single model to evaluate (e.g., 'ViT-B/32')")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path for single model evaluation")
    
    # Multiple model comparison args
    parser.add_argument("--conclip-b32-ckpt", type=str, default=None)
    parser.add_argument("--conclip-b16-ckpt", type=str, default=None)
    parser.add_argument("--conclip-l14-ckpt", type=str, default=None)
    
    # Output args
    parser.add_argument("--output", type=str, default="sugarcrepe_results.pt",
                        help="Path to save results")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    main(args)