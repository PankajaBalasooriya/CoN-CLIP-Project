import argparse, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BlipProcessor, BlipForImageTextRetrieval
from data import CCNegEvalDataset  # your dataset

def collate_fn(batch):
    images = [item[0][0] if isinstance(item[0], list) else item[0] for item in batch]
    true_captions = [item[1][0] for item in batch]
    negated_captions = [item[2][0] for item in batch]
    return images, true_captions, negated_captions


@torch.no_grad()
def main(args):
    device = args.device

    # ✅ Correct model for retrieval
    model_id = "Salesforce/blip-itm-base-coco"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForImageTextRetrieval.from_pretrained(model_id).to(device)
    model.eval()

    ds = CCNegEvalDataset(transform=None)
    loader = DataLoader(
        ds, batch_size=args.batch, num_workers=args.workers,
        collate_fn=collate_fn, pin_memory=(device == "cuda")
    )

    correct, total = 0, 0
    bar = tqdm(total=len(loader))

    for images, true_caps, neg_caps in loader:
        images = [img[0] if isinstance(img, list) else img for img in images]

        # Tokenize + preprocess both text and image
        inputs_true = processor(text=true_caps, images=images, return_tensors="pt", padding=True).to(device)
        inputs_neg  = processor(text=neg_caps,  images=images, return_tensors="pt", padding=True).to(device)

        # ✅ Extract embeddings manually
        # Image embeddings
        vision_outputs = model.vision_model(inputs_true.pixel_values)
        image_embeds = vision_outputs.last_hidden_state[:, 0, :]  # CLS token

        # Text embeddings (use CLS token instead of pooler)
        text_outputs_true = model.text_encoder(
            input_ids=inputs_true.input_ids,
            attention_mask=inputs_true.attention_mask,
            return_dict=True
        ).last_hidden_state[:, 0, :]

        text_outputs_neg = model.text_encoder(
            input_ids=inputs_neg.input_ids,
            attention_mask=inputs_neg.attention_mask,
            return_dict=True
        ).last_hidden_state[:, 0, :]

        # Normalize embeddings
        img_f = F.normalize(image_embeds, dim=-1)
        t_f   = F.normalize(text_outputs_true, dim=-1)
        n_f   = F.normalize(text_outputs_neg, dim=-1)

        # Compute cosine similarity
        sim_true = (img_f * t_f).sum(dim=-1, keepdim=True)
        sim_neg  = (img_f * n_f).sum(dim=-1, keepdim=True)

        sim = torch.cat([sim_true, sim_neg], dim=1)
        preds = sim.argmax(dim=-1)
        labels = torch.zeros(sim.size(0), device=device, dtype=preds.dtype)

        correct += (preds == labels).sum().item()
        total += sim.size(0)
        bar.set_postfix({"accuracy": round(correct / total * 100, 3)})
        bar.update(1)

    print(f"Final BLIP Base (ITM) CC-Neg accuracy: {correct / total * 100:.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    main(args)
