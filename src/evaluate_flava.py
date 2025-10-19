import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import FlavaProcessor, FlavaModel
from tqdm import tqdm
from data import CCNegEvalDataset  # your dataset class

# Collate function: returns images and captions directly
def collate_fn(batch):
    images = [item[0] for item in batch]
    true_captions = [item[1][0] for item in batch]
    negated_captions = [item[2][0] for item in batch]
    return images, true_captions, negated_captions


@torch.no_grad()
def main(args):
    device = args.device
    processor = FlavaProcessor.from_pretrained("facebook/flava-full")
    model = FlavaModel.from_pretrained("facebook/flava-full").to(device)
    model.eval()

    # Let processor handle image preprocessing
    ds = CCNegEvalDataset(transform=None)
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    correct, total = 0, 0
    bar = tqdm(total=len(loader))

    for images, true_caps, neg_caps in loader:
        # Prepare input batches
        inputs_true = processor(text=true_caps, images=images, return_tensors="pt", padding=True).to(device)
        inputs_neg = processor(text=neg_caps, images=images, return_tensors="pt", padding=True).to(device)

        # Forward pass through FLAVA
        out_true = model(**inputs_true)
        out_neg = model(**inputs_neg)

        # ✅ Mean-pool embeddings manually
        img_f = F.normalize(out_true.image_embeddings.mean(dim=1), dim=-1)
        t_f = F.normalize(out_true.text_embeddings.mean(dim=1), dim=-1)
        n_f = F.normalize(out_neg.text_embeddings.mean(dim=1), dim=-1)

        # Compute cosine similarities
        sim_true = (img_f * t_f).sum(dim=-1, keepdim=True)
        sim_neg = (img_f * n_f).sum(dim=-1, keepdim=True)

        # Compare similarities and compute accuracy
        sim = torch.cat([sim_true, sim_neg], dim=1)
        preds = sim.argmax(dim=-1)
        labels = torch.zeros(sim.size(0), device=device, dtype=preds.dtype)

        correct += (preds == labels).sum().item()
        total += sim.size(0)

        bar.set_postfix({"accuracy": round(correct / total * 100, 3)})
        bar.update(1)

    bar.close()
    print(f"\n✅ Final FLAVA CC-Neg Accuracy: {correct / total * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    main(args)
