# -*- coding: utf-8 -*-
"""
Batch verification for SCABG_LDT_Model.

Features
- Read dataset: .json / .jsonl / .csv (utf-8)
- Fields: index, Label, ImageID, Description, Entity
- Replace "$T" with Entity in Description
- Load images from --image_root/ImageID
- Batched inference on GPU/CPU
- Save per-sample outputs (pred, probs, correctness) to CSV/JSON
- Print summary metrics (accuracy, confusion matrix)

Usage:
python batch_infer.py \
  --ckpt path/to/ckpt.pt \
  --data path/to/dataset.json \
  --image_root /path/to/images \
  --out results.csv \
  --batch_size 32 \
  --max_len 128
"""
import os, json, csv, argparse, math
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from multimodal_mh_upgraded import SCABG_LDT_Model, UpgConfig


# -----------------------------
# Utils
# -----------------------------
def load_checkpoint(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    cfg_dict = ckpt.get("cfg", {})
    if not isinstance(cfg_dict, dict):
        cfg_dict = {}
    cfg = UpgConfig()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    model = SCABG_LDT_Model(cfg).to(device)
    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except Exception:
        model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # Try to infer num_labels if present; fallback to 2
    num_labels = getattr(cfg, "num_labels", None)
    if num_labels is None:
        # best-effort: from classifier head if exposed
        try:
            # assume model.classifier or similar
            if hasattr(model, "classifier"):
                num_labels = model.classifier.out_features
            else:
                num_labels = 2
        except Exception:
            num_labels = 2

    return model, cfg, num_labels


def read_any_dataset(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                return data["data"]
            else:
                raise ValueError("JSON must be a list of records or contain a top-level 'data' list.")
    elif ext == ".csv":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        return rows
    else:
        raise ValueError(f"Unsupported data extension: {ext}")


# -----------------------------
# Dataset
# -----------------------------
class MMSentDataset(Dataset):
    def __init__(self,
                 records: List[Dict[str, Any]],
                 image_root: str,
                 tokenizer: AutoTokenizer,
                 max_len: int = 128,
                 use_ctx: bool = False):
        self.records = records
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_ctx = use_ctx

        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def _encode(self, text: str):
        o = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return o["input_ids"].squeeze(0), o["attention_mask"].squeeze(0)

    def __getitem__(self, i):
        r = self.records[i]
        idx = str(r.get("index", i))
        label_str = r.get("Label", "")
        try:
            label = int(label_str)
        except Exception:
            label = -1  # unknown label

        img_name = r.get("ImageID", "")
        img_path = os.path.join(self.image_root, img_name)

        # Compose text by replacing $T with Entity
        desc = r.get("Description", "")
        entity = r.get("Entity", "")
        text = desc.replace("$T", str(entity)) if entity is not None else desc

        # Optional context (if you have a field, else keep empty)
        ctx_text = r.get("Ctx", "")  # not in your sample; kept for compatibility

        # Load image
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.tf(img)
        except Exception:
            # create a blank image tensor if missing/corrupt
            img = torch.zeros(3, 224, 224, dtype=torch.float32)

        # Tokenize
        n_ids, n_mask = self._encode(text)
        if self.use_ctx and isinstance(ctx_text, str) and len(ctx_text) > 0:
            c_ids, c_mask = self._encode(ctx_text)
        else:
            c_ids = None
            c_mask = None

        return {
            "index": idx,
            "label": label,
            "image": img,
            "n_ids": n_ids,
            "n_mask": n_mask,
            "c_ids": c_ids,
            "c_mask": c_mask,
            "image_path": img_path,
            "text_used": text,
            "entity": entity,
        }


def collate_fn(batch: List[Dict[str, Any]]):
    # images
    imgs = torch.stack([b["image"] for b in batch], dim=0)
    # text ids/masks
    n_ids = torch.stack([b["n_ids"] for b in batch], dim=0)
    n_mask = torch.stack([b["n_mask"] for b in batch], dim=0)
    # ctx (may be None)
    if batch[0]["c_ids"] is None:
        c_ids = None
        c_mask = None
    else:
        c_ids = torch.stack([b["c_ids"] for b in batch], dim=0)
        c_mask = torch.stack([b["c_mask"] for b in batch], dim=0)

    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    indices = [b["index"] for b in batch]
    img_paths = [b["image_path"] for b in batch]
    texts = [b["text_used"] for b in batch]
    entity = [b["entity"] for b in batch]

    return {
        "images": imgs,
        "n_ids": n_ids,
        "n_mask": n_mask,
        "c_ids": c_ids,
        "c_mask": c_mask,
        "labels": labels,
        "indices": indices,
        "img_paths": img_paths,
        "texts": texts,
        "entity": entity,
    }


# -----------------------------
# Metrics
# -----------------------------
def confusion_matrix(preds: List[int], labels: List[int], num_labels: int):
    cm = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    for p, y in zip(preds, labels):
        if 0 <= y < num_labels and 0 <= p < num_labels:
            cm[y][p] += 1
    return cm


def accuracy(preds: List[int], labels: List[int]):
    correct = sum(int(p == y) for p, y in zip(preds, labels) if y >= 0)
    total = sum(1 for y in labels if y >= 0)
    return (correct / total) if total > 0 else 0.0


# -----------------------------
# Save helpers
# -----------------------------
def save_csv(out_path: str, rows: List[Dict[str, Any]]):
    keys = sorted({k for r in rows for k in r.keys()})
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_json(out_path: str, rows: List[Dict[str, Any]]):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str, help="Path to checkpoint .pt")
    ap.add_argument("--data", required=True, type=str, help="Path to dataset (.json/.jsonl/.csv)")
    ap.add_argument("--image_root", required=True, type=str, help="Directory containing images")
    ap.add_argument("--out", required=True, type=str, help="Output file (.csv or .json)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--ctx", action="store_true", help="Use ctx_text if available (field 'Ctx')")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, num_labels = load_checkpoint(args.ckpt, device)

    tokenizer = AutoTokenizer.from_pretrained(getattr(cfg, "roberta_name", "roberta-large"), use_fast=True)

    records = read_any_dataset(args.data)
    ds = MMSentDataset(records, args.image_root, tokenizer, max_len=args.max_len, use_ctx=args.ctx)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )

    all_rows = []
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dl:
            imgs = batch["images"].to(device)
            n_ids = batch["n_ids"].to(device)
            n_mask = batch["n_mask"].to(device)
            c_ids = batch["c_ids"].to(device) if batch["c_ids"] is not None else None
            c_mask = batch["c_mask"].to(device) if batch["c_mask"] is not None else None
            labels = batch["labels"].tolist()
            # entity = batch["entities"]

            out = model(imgs, n_ids, n_mask, c_ids, c_mask, labels=None)
            logits = out["logits"]  # (B, C)
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            preds = torch.argmax(logits, dim=-1).cpu().tolist()

            for i in range(len(preds)):
                row = {
                    "index": batch["indices"][i],
                    "image_path": batch["img_paths"][i],
                    "text": batch["texts"][i],
                    "label": labels[i],
                    "pred": preds[i],
                    "entity": batch["entity"][i],
                    "correct": int(labels[i] == preds[i]) if labels[i] >= 0 else "",
                    "probs": json.dumps(probs[i], ensure_ascii=False),
                }
                all_rows.append(row)

            all_preds.extend(preds)
            all_labels.extend(labels)

    # Metrics
    acc = accuracy(all_preds, all_labels)
    cm = confusion_matrix(all_preds, all_labels, num_labels=num_labels)

    # Save outputs
    out_ext = os.path.splitext(args.out)[1].lower()
    if out_ext == ".csv":
        save_csv(args.out, all_rows)
    elif out_ext == ".json":
        save_json(args.out, all_rows)
    else:
        # default to CSV
        save_csv(args.out, all_rows)

    # Print summary
    print(f"# Samples: {len(all_rows)}")
    print(f"Accuracy: {acc*100:.2f}%")
    print("Confusion Matrix (rows = true, cols = pred):")
    for r in cm:
        print("  " + "\t".join(str(x) for x in r))

    # Also write a small summary next to out file
    summ_path = os.path.splitext(args.out)[0] + "_summary.txt"
    with open(summ_path, "w", encoding="utf-8") as f:
        f.write(f"# Samples: {len(all_rows)}\n")
        f.write(f"Accuracy: {acc*100:.2f}%\n")
        f.write("Confusion Matrix (rows = true, cols = pred):\n")
        for r in cm:
            f.write("  " + "\t".join(str(x) for x in r) + "\n")


if __name__ == "__main__":
    main()
