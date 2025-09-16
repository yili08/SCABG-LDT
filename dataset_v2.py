# -*- coding: utf-8 -*-
import os, json, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer

def _read_table_any(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".json") or p.endswith(".jsonl"):
        
        try:
            return pd.read_json(path, lines=True)
        except Exception:
            return pd.read_json(path)
    else:
        return pd.read_csv(path)

def _json_to_tensor(val):
    if val is None: return None
    if isinstance(val, (list, tuple)):
        arr = torch.tensor(val, dtype=torch.float32)
        return arr if arr.numel() > 0 else None
    if isinstance(val, str):
        s = val.strip()
        if not s.startswith("["): return None
        try:
            arr = torch.tensor(json.loads(s), dtype=torch.float32)
            return arr if arr.numel() > 0 else None
        except Exception:
            return None
    return None

def build_transform(img_size=224, img_norm="imagenet"):
    
   
    
    if img_norm == "vit":
        mean = [0.5, 0.5, 0.5]
        std  = [0.5, 0.5, 0.5]
    else:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

class FNDatasetV2(Dataset):
    
    
    def __init__(self, csv_path, image_root="", roberta_name="roberta-large",
                 max_len=128, use_ctx=True, train=True, img_size=224, img_norm="imagenet"):
        self.df = _read_table_any(csv_path)
        self.image_root = image_root
        self.tokenizer = AutoTokenizer.from_pretrained(roberta_name, use_fast=True, cache_dir="models")
        self.max_len = max_len
        self.use_ctx = use_ctx
        self.train = train

        
        for need in ["image_path", "text"]:
            if need not in self.df.columns:
                raise KeyError(f"Missing column: {need}")
        if "ctx_text" not in self.df.columns: self.df["ctx_text"] = ""
        if "label" not in self.df.columns and "Label" in self.df.columns:
            self.df["label"] = self.df["Label"]
        if "label" not in self.df.columns:
            self.df["label"] = -1

        self.t = build_transform(img_size=img_size, img_norm=img_norm)

    def __len__(self):
        return len(self.df)

    def _tok(self, text: str):
        return self.tokenizer(
            str(text) if isinstance(text, str) else "",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # image
        ip = str(row["image_path"])
        if self.image_root and not os.path.isabs(ip):
            ip = os.path.join(self.image_root, ip)
        image = Image.open(ip).convert("RGB")
        image = self.t(image)

        # text
        news = self._tok(row["text"])
        item = {
            "image": image,
            "news_ids": news["input_ids"].squeeze(0),
            "news_mask": news["attention_mask"].squeeze(0),
        }

        # context 
        if self.use_ctx:
            ctx = self._tok(row.get("ctx_text", ""))
            item["ctx_ids"]  = ctx["input_ids"].squeeze(0)
            item["ctx_mask"] = ctx["attention_mask"].squeeze(0)

        
        if self.train and "label" in row:
            item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)

        
        t_tf = _json_to_tensor(row.get("teacher_tf"))
        t_it = _json_to_tensor(row.get("teacher_it"))
        if t_tf is not None: item["teacher_tf"] = t_tf
        if t_it is not None: item["teacher_it"] = t_it

        
        c = row.get("coarse_label")
        if c is not None:
            try:
                item["coarse"] = torch.tensor(int(c), dtype=torch.long)
            except Exception:
                pass

        return item
