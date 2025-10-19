"""
Zero-shot Image Classification Evaluation for CLIP and CoN-CLIP
Evaluates models on: ImageNet-1k, Caltech-101, Flowers-102, CIFAR-100, 
Food-101, Stanford Cars, Oxford Pets, CIFAR-10
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


# ==================== Dataset Class Names ====================

IMAGENET_CLASSES = None  # Load from file or use subset


# CALTECH101_CLASSES = [
#     "off-center face", "centered face", "leopard", "motorbike", "accordion",
#     "airplane", "anchor", "ant", "barrel", "bass", "beaver", "binocular",
#     "bonsai", "brain", "brontosaurus", "buddha", "butterfly", "camera",
#     "cannon", "side of a car", "ceiling fan", "cellphone", "chair", "chandelier",
#     "body of a cougar cat", "face of a cougar cat", "crab", "crayfish",
#     "crocodile", "head of a crocodile", "cup", "dalmatian", "dollar bill",
#     "dolphin", "dragonfly", "electric guitar", "elephant", "emu", "euphonium",
#     "ewer", "ferry", "flamingo", "head of a flamingo", "garfield", "gerenuk",
#     "gramophone", "grand piano", "hawksbill", "headphone", "hedgehog",
#     "helicopter", "ibis", "inline skate", "joshua tree", "kangaroo", "ketch",
#     "lamp", "laptop", "llama", "lobster", "lotus", "mandolin", "mayfly",
#     "menorah", "metronome", "minaret", "nautilus", "octopus", "okapi",
#     "pagoda", "panda", "pigeon", "pizza", "platypus", "pyramid", "revolver",
#     "rhino", "rooster", "saxophone", "schooner", "scissors", "scorpion",
#     "sea horse", "snoopy (cartoon beagle)", "soccer ball", "stapler",
#     "starfish", "stegosaurus", "stop sign", "strawberry", "sunflower",
#     "tick", "trilobite", "umbrella", "watch", "water lilly", "wheelchair",
#     "wild cat", "windsor chair", "wrench", "yin and yang symbol"
# ]

CALTECH101_CLASSES = [
    "accordion", "chandelier", "flamingo_head", "mayfly", "sea_horse",
    "airplanes", "cougar_body", "garfield", "menorah", "snoopy",
    "anchor", "cougar_face", "gerenuk", "metronome", "soccer_ball",
    "ant", "crab", "gramophone", "minaret", "stapler",
    "BACKGROUND_Google", "crayfish", "grand_piano", "Motorbikes", "starfish",
    "barrel", "crocodile", "hawksbill", "nautilus", "stegosaurus",
    "bass", "crocodile_head", "headphone", "octopus", "stop_sign",
    "beaver", "cup", "hedgehog", "okapi", "strawberry",
    "binocular", "dalmatian", "helicopter", "pagoda", "sunflower",
    "bonsai", "dollar_bill", "ibis", "panda", "tick",
    "brain", "dolphin", "inline_skate", "pigeon", "trilobite",
    "brontosaurus", "dragonfly", "joshua_tree", "pizza", "umbrella",
    "buddha", "electric_guitar", "kangaroo", "platypus", "watch",
    "butterfly", "elephant", "ketch", "pyramid", "water_lilly",
    "caltech101", "emu", "lamp", "revolver", "wheelchair",
    "camera", "euphonium", "laptop", "rhino", "wild_cat",
    "cannon", "ewer", "Leopards", "rooster", "windsor_chair",
    "car_side", "Faces", "llama", "saxophone", "wrench",
    "ceiling_fan", "Faces_easy", "lobster", "schooner", "yin_yang",
    "cellphone", "ferry", "lotus", "scissors", "chair",
    "flamingo", "mandolin", "scorpion"
]

FLOWERS102_CLASSES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
    "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
    "stemless gentian", "artichoke", "sweet william", "carnation", "garden phlox",
    "love in the mist", "mexican aster", "alpine sea holly", "ruby-lipped cattleya",
    "cape flower", "great masterwort", "siam tulip", "lenten rose", "barbeton daisy",
    "daffodil", "sword lily", "poinsettia", "bolero deep blue", "wallflower",
    "marigold", "buttercup", "oxeye daisy", "common dandelion", "petunia",
    "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff",
    "gaura", "geranium", "orange dahlia", "pink-yellow dahlia", "cautleya spicata",
    "japanese anemone", "black-eyed susan", "silverbush", "californian poppy",
    "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy",
    "gazania", "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily", "anthurium", "frangipani", "clematis",
    "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia", "cyclamen",
    "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove",
    "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily"
]

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]

FOOD101_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots",
    "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries",
    "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna", "lobster_bisque",
    "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup", "mussels",
    "nachos", "omelette", "onion_rings", "oysters", "pad_thai", "paella", "pancakes",
    "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak",
    "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
    "waffles"
]

STANFORD_CARS_CLASSES = None  # 196 car models - load from file

OXFORD_PETS_CLASSES = [
    "Abyssinian", "American Bulldog", "American Pit Bull Terrier", "Basset Hound",
    "Beagle", "Bengal", "Birman", "Bombay", "Boxer", "British Shorthair",
    "Chihuahua", "Egyptian Mau", "English Cocker Spaniel", "English Setter",
    "German Shorthaired", "Great Pyrenees", "Havanese", "Japanese Chin", "Keeshond",
    "Leonberger", "Maine Coon", "Miniature Pinscher", "Newfoundland", "Persian",
    "Pomeranian", "Pug", "Ragdoll", "Russian Blue", "Saint Bernard", "Samoyed",
    "Scottish Terrier", "Shiba Inu", "Siamese", "Sphynx", "Staffordshire Bull Terrier",
    "Wheaten Terrier", "Yorkshire Terrier"
]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ==================== Template Functions ====================

def get_prompts(classnames, template="a photo of a {}."):
    """Generate text prompts for each class."""
    return [template.format(c.replace("_", " ")) for c in classnames]


# ==================== Dataset Loading Functions ====================

# def load_imagenet(root, preprocess, batch_size=256):
#     """Load ImageNet validation set."""
#     dataset = datasets.ImageNet(
#         root=root,
#         split='val',
#         transform=preprocess
#     )
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
#     # Load class names
#     with open(os.path.join(root, 'imagenet_classes.txt'), 'r') as f:
#         classes = [line.strip() for line in f.readlines()]
    
#     return loader, classes


# def load_caltech101(root, preprocess, batch_size=256):
#     """Load Caltech-101."""
#     print(root)
#     dataset = datasets.Caltech101(
#         root=root,
#         target_type='category',
#         transform=preprocess,
#         download=False
#     )
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
#     return loader, CALTECH101_CLASSES

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

    # ✅ Automatically get class names in the correct order
    classnames = dataset.classes  # this returns 102 flower class names in correct order
    
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
    
    # ✅ Get class names directly from the dataset
    classnames = dataset.classes  # list of 100 fine classes
    
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
    return loader, FOOD101_CLASSES


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
    """Load Oxford-IIIT Pets."""
    dataset = datasets.OxfordIIITPet(
        root=root,
        split='test',
        transform=preprocess,
        download=True
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return loader, OXFORD_PETS_CLASSES


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
    
    # Use official class labels directly from the dataset
    classnames = dataset.classes  # ['airplane', 'automobile', 'bird', ..., 'truck']
    
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


# ==================== Main Evaluation ====================

def load_model(model_name, checkpoint_path=None, device="cuda"):
    """Load CLIP or CoN-CLIP model."""

    model, preprocess = clip.load(model_name, device=device)
    # Extract actual CLIP model name (the part after the first hyphen)


    
    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model = model.float()
        model.load_state_dict(ckpt["model"])
    
    model = model.to(device)
    model.eval()
    
    return model, preprocess



def evaluate_all_datasets(model, preprocess, data_root, device="cuda"):
    """Evaluate model on all datasets."""
    results = {}
    
    datasets_config = {"CIFAR-100": (load_cifar100, os.path.join(data_root, "cifar100")),
        
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


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Configuration
    DATA_ROOT = "/home/pankaja/datasets"  # Change this
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Models to evaluate
    # models_config = {
    #     "CLIP-ViT-B/32": None,
    #     "CoN-CLIP-ViT-B/32": "/home/pankaja/ENTC/Sem5/CoN-CLIP-Project/logs/conclip_b32/results_conclip_b32.pt",
    #     "CLIP-ViT-B/16": None,
    #     "CoN-CLIP-ViT-B/16": "/home/pankaja/ENTC/Sem5/CoN-CLIP-Project/logs/conclip_b16/results_conclip_b16.pt",
    #     "CLIP-ViT-L/14": None
    # }

    models_config = {
        "ViT-B/32": None,
        "ViT-B/16": None,
    }
    
    # Run evaluation
    results = run_comparison(models_config, DATA_ROOT, DEVICE)
    
    # Print results
    print_results_table(results)
    
    # Save results
    torch.save(results, "zero_shot_results.pt")
    print("\nResults saved to zero_shot_results.pt")