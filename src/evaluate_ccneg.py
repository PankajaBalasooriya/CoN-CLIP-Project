import argparse, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import clip
from data import CCNegEvalDataset
from tqdm import tqdm

def collate_fn(batch):
    images = [item[0] for item in batch]
    true_captions = [item[1][0] for item in batch]
    negated_captions = [item[2][0] for item in batch]
    return (torch.stack(images, 0), true_captions, negated_captions)

@torch.no_grad()
def main(args):
    device = args.device
    model, preprocess = clip.load(args.model, device=device)
    
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        model = model.float()
        model.load_state_dict(ckpt["model"])
        model = model.to(device)


    ds = CCNegEvalDataset(transform=preprocess)  # looks in ccneg_dataset/ by default
    loader = DataLoader(
        ds, batch_size=args.batch, pin_memory=(device=="cuda"),
        num_workers=args.workers, collate_fn=collate_fn
    )
    bar = tqdm(total=len(loader))
    correct, total = 0, 0

    for (images, true_caps, neg_caps) in loader:
        images = images.to(device, dtype=getattr(model, "dtype", torch.float32))
        t_tok = clip.tokenize(true_caps).to(device)
        n_tok = clip.tokenize(neg_caps).to(device)

        img_f = F.normalize(model.encode_image(images), dim=-1)
        t_f   = F.normalize(model.encode_text(t_tok), dim=-1)
        n_f   = F.normalize(model.encode_text(n_tok), dim=-1)

        # Either scaling works for argmax; use .exp() to mirror CLIP forward
        scale = model.logit_scale.exp() if hasattr(model, "logit_scale") else 1.0

        # Compute only the diagonal similarities (avoid BxB memory)
        sim_true = (img_f * t_f).sum(dim=-1, keepdim=True) * scale
        sim_neg  = (img_f * n_f).sum(dim=-1, keepdim=True) * scale

        sim = torch.cat([sim_true, sim_neg], dim=1)
        preds = sim.argmax(dim=-1)
        labels = torch.zeros(sim.size(0), device=device, dtype=preds.dtype)  # true captions are column 0

        correct += (preds == labels).sum().item()
        total += sim.size(0)
        bar.set_postfix({"accuracy": round(correct/total * 100, 3)})
        bar.update(1)

    print(f"Final CC-Neg accuracy: {correct/total*100:.2f}%")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ViT-B/32")
    p.add_argument("--ckpt", default="", help="Optional CoN-CLIP .pt path")
    p.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    main(args)
