"""
Zero-shot Image Classification Evaluation for CLIP and CoN-CLIP
Evaluates models on: Caltech-101, Flowers-102, CIFAR-100, Oxford Pets, CIFAR-10
"""

import torch
import torch.nn.functional as F
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import json
import numpy as np

# ==================== Template Functions ====================

def get_prompts(classnames, template="a photo of a {}."):
    """Generate text prompts for each class."""
    return [template.format(c.replace("_", " ")) for c in classnames]



def load_caltech101(root, preprocess, batch_size=256):
    """Load Caltech-101."""
    dataset = datasets.Caltech101(
        root=root,
        target_type='category',
        transform=preprocess,
        download=False
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Get the actual class names in correct order
    classnames = dataset.categories
    
    return loader, classnames



def load_flowers102(root, preprocess, batch_size=256):
    """Load Flowers-102 with correct class labels from the dataset."""
    dataset = datasets.Flowers102(
        root=root,
        split='test',
        transform=preprocess,
        download=True
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    classnames = dataset.classes  
    
    return loader, classnames



def load_cifar100(root, preprocess, batch_size=256):
    """Load CIFAR-100 with correct class names."""
    dataset = datasets.CIFAR100(
        root=root,
        train=False,
        transform=preprocess,
        download=True
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    classnames = dataset.classes 
    
    return loader, classnames



def load_food101(root, preprocess, batch_size=256):
    """Load Food-101."""
    dataset = datasets.Food101(
        root=root,
        split='test',
        transform=preprocess,
        download=True
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # return loader, FOOD101_CLASSES
    classnames = dataset.classes  
    return loader, classnames


def load_stanford_cars(root, preprocess, batch_size=256):
    """Load Stanford Cars."""
    dataset = datasets.StanfordCars(
        root=root,
        split='test',
        transform=preprocess,
        download=True
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load class names from metadata
    import scipy.io
    meta = scipy.io.loadmat(os.path.join(root, 'stanford_cars', 'cars_meta.mat'))
    classes = [c[0] for c in meta['class_names'][0]]
    
    return loader, classes


def load_oxford_pets(root, preprocess, batch_size=256):
    """Load Oxford-IIIT Pets with class names directly from the dataset."""
    dataset = datasets.OxfordIIITPet(
        root=root,
        split='test',
        transform=preprocess,
        download=True
    )
    
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    classnames = dataset.classes

    return loader, classnames



def load_cifar10(root, preprocess, batch_size=256):
    """Load CIFAR-10 dataset with correct class labels from torchvision."""
    dataset = datasets.CIFAR10(
        root=root,
        train=False,
        transform=preprocess,
        download=True
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    classnames = dataset.classes 
    
    return loader, classnames



# ==================== Evaluation Function ====================

@torch.no_grad()
def evaluate_zero_shot(model, loader, text_features, device):
    """
    Evaluate zero-shot classification accuracy.
    
    Args:
        model: CLIP model
        loader: DataLoader for the dataset
        text_features: Pre-computed text features for all classes
        device: torch device
    
    Returns:
        top1_accuracy: Top-1 accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)
        
        # Encode images
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        
        # Compute similarity
        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Get predictions
        predictions = logits.argmax(dim=-1)
        
        # Update accuracy
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    accuracy = 100 * correct / total
    return accuracy


def compute_text_features(model, classnames, device, template="a photo of a {}."):
    """Compute text features for all classes."""
    prompts = get_prompts(classnames, template)
    text_tokens = clip.tokenize(prompts).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)
    
    return text_features


# ===================== Loading the model ==========================
def load_model(model_name, checkpoint_path=None, device="cuda"):
    """Load CLIP or CoN-CLIP model."""

    # Extract actual CLIP model name (remove prefixes)
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
        print("Done Loading Checkpoint")
        model = model.float()
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print("Checkpoint type:", type(ckpt))
        # if isinstance(ckpt, dict):
        #     print("Checkpoint keys:", ckpt.keys())
        # else:
        #     print("Not a dict — raw object loaded.")

        if "model" in ckpt:
            print("Loading state dict from 'model' key in checkpoint")
            model.load_state_dict(ckpt["model"])
        else:
            print("Loading state dict directly from checkpoint")
            model.load_state_dict(ckpt)
    

    model = model.to(device)
    model.eval()
    return model, preprocess


def evaluate_all_datasets(model, preprocess, data_root, device="cuda"):
    """Evaluate model on all datasets."""
    results = {}
    
    datasets_config = {"Caltech-101": (load_caltech101, os.path.join(data_root, "caltech101")), 
                        "CIFAR-10": (load_cifar10, os.path.join(data_root, "cifar10")),
                        "Flowers-102": (load_flowers102, os.path.join(data_root, "flowers102")),
                        "CIFAR-100": (load_cifar100, os.path.join(data_root, "cifar100")),
                        "Oxford Pets": (load_oxford_pets, os.path.join(data_root, "oxford_pets")),
    }
    # datasets_config = {
    #     "Caltech-101": (load_caltech101, os.path.join(data_root, "caltech101")), 
    #     "Flowers-102": (load_flowers102, os.path.join(data_root, "flowers102")),
    #     "CIFAR-100": (load_cifar100, os.path.join(data_root, "cifar100")),
    #     "Food-101": (load_food101, os.path.join(data_root, "food101")),
    #     "Stanford Cars": (load_stanford_cars, os.path.join(data_root, "stanford_cars")),
    #     "Oxford Pets": (load_oxford_pets, os.path.join(data_root, "oxford_pets")),
    #     "CIFAR-10": (load_cifar10, os.path.join(data_root, "cifar10")),
    # }
    
    for dataset_name, (load_fn, dataset_path) in datasets_config.items():
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name}")
        print(f"{'='*60}")
        
        try:
            # Load dataset
            loader, classnames = load_fn(dataset_path, preprocess)
            
            # Compute text features
            text_features = compute_text_features(model, classnames, device)
            
            # Evaluate
            accuracy = evaluate_zero_shot(model, loader, text_features, device)
            
            results[dataset_name] = accuracy
            print(f"{dataset_name}: {accuracy:.2f}%")
            
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")
            results[dataset_name] = None
    
    return results


def run_comparison(models_config, data_root, device="cuda"):
    """
    Run comparison across multiple models.
    
    Args:
        models_config: Dict of {model_name: checkpoint_path}
        data_root: Root directory containing all datasets
        device: torch device
    """
    all_results = {}
    
    for model_name, checkpoint_path in models_config.items():
        print(f"\n{'#'*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'#'*60}")
        
        # Load model
        model, preprocess = load_model(
            model_name,
            checkpoint_path,
            device
        )
        
        # Evaluate on all datasets
        results = evaluate_all_datasets(model, preprocess, data_root, device)
        all_results[model_name] = results
    
    return all_results


def print_results_table(results):
    """Print results in a formatted table."""
    print("\n" + "="*80)
    print("ZERO-SHOT IMAGE CLASSIFICATION RESULTS")
    print("="*80)
    
    # Get dataset names
    datasets = list(next(iter(results.values())).keys())
    
    # Print header
    print(f"{'Model':<25}", end="")
    for dataset in datasets:
        short_name = dataset.replace("-", "").replace(" ", "")[:8]
        print(f"{short_name:>10}", end="")
    print(f"{'Avg':>10}")
    print("-"*80)
    
    # Print results for each model
    for model_name, model_results in results.items():
        print(f"{model_name:<25}", end="")
        accuracies = []
        for dataset in datasets:
            acc = model_results[dataset]
            if acc is not None:
                print(f"{acc:>10.2f}", end="")
                accuracies.append(acc)
            else:
                print(f"{'N/A':>10}", end="")
        
        if accuracies:
            avg_acc = np.mean(accuracies)
            print(f"{avg_acc:>10.2f}")
        else:
            print(f"{'N/A':>10}")
    
    print("="*80)



if __name__ == "__main__":
    # Configuration
    DATA_ROOT = "/home/pankaja/datasets"  
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Models to evaluate
    # models_config = {
    #     "CLIP-ViT-B/32": None,
    #     "CoN-CLIP-ViT-B/32": "/home/pankaja/ENTC/Sem5/CoN-CLIP-Project/checkpoints/conclip_b32/ckpt_5_conclip_b32.pt",
    #     "CLIP-ViT-B/16": None,
    #     "CoN-CLIP-ViT-B/16": "/home/pankaja/ENTC/Sem5/CoN-CLIP-Project/checkpoints/conclip_b16/ckpt_5_conclip_b16.pt",
    #     "CLIP-ViT-L/14": None,
    #     "CoN-CLIP-ViT-L/14": "/home/pankaja/ENTC/Sem5/CoN-CLIP-Project/checkpoints/conclip_l14/ckpt_5_conclip_l14.pt",
    # }

    models_config = {
        "CLIP-ViT-B/32": None,
        "CoN-CLIP-ViT-B/32": "/home/pankaja/ENTC/Sem5/CoN-CLIP-Project/checkpoints/conclip_b32/ckpt_5_conclip_b32.pt",
    }

    # Run evaluation
    results = run_comparison(models_config, DATA_ROOT, DEVICE)
    
    # Print results
    print_results_table(results)
    
    # Save results
    torch.save(results, "zero_shot_results.pt")
    print("\nResults saved to zero_shot_results.pt")