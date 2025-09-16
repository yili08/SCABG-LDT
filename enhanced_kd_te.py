# -*- coding: utf-8 -*-
"""
Precompute teacher logits for KD (Entity-centric, binary tasks)
---------------------------------------------------------------
Teacher 1 (Textual NLI -> Emotion presence w.r.t Entity, binary):
  Input  : premise = [Description with $T -> Entity] + [ctx_text]
           hypotheses:
             H_yes: "The mention of <Entity> conveys an opinion or sentiment."
             H_no : "The mention of <Entity> is purely objective and contains no opinion."
  Output : teacher_tf = [p(no-emotion), p(emotion)]   # 2-dim JSON string

Supervised calibration (label-aware bias):
  Orig labels: 0=neg, 1=neutral, 2=pos
  Coarse for this stage: neutral(1) -> no-emotion(0); neg(0)/pos(2) -> emotion(1)
  We add a small bias to the corresponding class logit before softmax to reduce neutral-collapse.

Teacher 2 (CLIP ITM -> Image-Desc-Entity match, binary):
  Text  : "Desc($T->Entity). The entity is: <Entity>."
  Output: teacher_it = [p(unmatch), p(match)]         # 2-dim JSON string

Notes:
- Supports local caching via --model_dir, avoids re-downloads.
- Robust IO for CSV/TSV/JSON/JSONL; output format inferred from --out extension.
- Batched inference for both NLI and CLIP.
"""

import argparse, os, json, math
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    CLIPProcessor, CLIPModel
)

# -------------------------
# Utils
# -------------------------
def softmax(x, dim=-1):
    return torch.softmax(x, dim=dim)

def load_any(path, sep='auto'):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".json", ".jsonl", ".jsn"]:
        # try JSON lines
        try:
            return pd.read_json(path, lines=True)
        except Exception:
            pass
        # try JSON array
        try:
            return pd.read_json(path)
        except Exception:
            pass
        # fallback: manual line-by-line json objects
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().rstrip(",")
                if s.startswith("{") and s.endswith("}"):
                    try:
                        rows.append(json.loads(s))
                    except Exception:
                        continue
        if rows:
            return pd.DataFrame(rows)
        raise ValueError(f"Could not parse JSON/JSONL file: {path}")

    if sep == 'auto':
        for try_sep in [None, "\t", ",", ";", "|"]:
            try:
                return pd.read_csv(path, sep=try_sep, engine="python", encoding="utf-8-sig")
            except Exception:
                continue
        # last resort
        return pd.read_csv(path, encoding="utf-8-sig")
    else:
        return pd.read_csv(path, sep=sep, engine="python", encoding="utf-8-sig")

def save_any(df, out_path):
    ext = os.path.splitext(out_path)[1].lower()
    if ext in [".json", ".jsonl", ".jsn"]:
        # write JSON lines
        recs = df.to_dict(orient="records")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        df.to_csv(out_path, index=False, encoding="utf-8")

def normalize_df(df):
    """Map your columns to the internal names the rest of the code expects."""
    lower = {c.lower(): c for c in df.columns}

    # Expected final columns: image_path, text, entity, ctx_text, label
    def pick(*cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    # image_path
    image_col = pick("image_path", lower.get("imageid"), lower.get("image_id"))
    if image_col is None:
        raise KeyError("Missing image column (expected ImageID or image_path).")
    if "image_path" not in df.columns:
        df["image_path"] = df[image_col].astype(str)

    # text (Description)
    text_col = pick("text", lower.get("description"), lower.get("desc"))
    if text_col is None:
        raise KeyError("Missing text column (expected Description or text).")
    if "text" not in df.columns:
        df["text"] = df[text_col].fillna("").astype(str)

    # entity
    entity_col = pick("entity", lower.get("entity"))
    if entity_col is None:
        raise KeyError("Missing entity column (expected Entity).")
    if "entity" not in df.columns:
        df["entity"] = df[entity_col].fillna("").astype(str)

    # ctx_text (LLM-generated image caption)
    ctx_col = pick("ctx_text", lower.get("ctx_text"), lower.get("ctx"))
    if "ctx_text" not in df.columns and ctx_col is not None:
        df["ctx_text"] = df[ctx_col].fillna("").astype(str)
    if "ctx_text" not in df.columns:
        df["ctx_text"] = ""

    # label (supervised original 0/1/2)
    lbl_col = pick("label", lower.get("label"), lower.get("Label"))
    if "label" not in df.columns and lbl_col is not None:
        df["label"] = df[lbl_col]
    if "label" not in df.columns:
        raise KeyError("Missing label column (expected Label/label).")

    # fill types
    df["image_path"] = df["image_path"].astype(str)
    df["text"] = df["text"].astype(str)
    df["entity"] = df["entity"].astype(str)
    df["ctx_text"] = df["ctx_text"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(1).astype(int)

    return df

# def ensure_cache(model_name, cache_root):
#     """Prime local cache; always use model_name + cache_dir when loading."""
#     os.makedirs(cache_root, exist_ok=True)
#     if "clip" in model_name.lower():
#         CLIPProcessor.from_pretrained(model_name, cache_dir=cache_root)
#         CLIPModel.from_pretrained(model_name, cache_dir=cache_root)
#     else:
#         AutoTokenizer.from_pretrained(model_name, cache_dir=cache_root)
#         AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_root)
#     return cache_root



def ensure_cache(model_name, cache_root):
    os.makedirs(cache_root, exist_ok=True)
    
    if os.path.isdir(model_name) and os.path.isfile(os.path.join(model_name, "config.json")):
        return model_name
    try:
        
        if "clip" in model_name.lower():
            CLIPProcessor.from_pretrained(model_name, cache_dir=cache_root, local_files_only=False)
            CLIPModel.from_pretrained(model_name, cache_dir=cache_root, local_files_only=False)
        else:
            AutoTokenizer.from_pretrained(model_name, cache_dir=cache_root, local_files_only=False)
            AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_root, local_files_only=False)
        return model_name  #  repo id + cache_dir
    except Exception:
        
        local_dir = os.path.join(cache_root, "snapshot")
        snapshot_download(repo_id=model_name, local_dir=local_dir,
                          local_dir_use_symlinks=False, resume_download=True)
        return local_dir  #  from_pretrained(local_dir)


def replace_T(desc, entity):
    """Replace $T with entity; keep original if no $T."""
    try:
        return desc.replace("$T", entity).strip()
    except Exception:
        return str(desc)

def to_json_list(x):
    try:
        return json.dumps([float(v) for v in x])
    except Exception:
        return "[]"

# -------------------------
# Teacher 1: NLI -> Emotion presence (binary)
# -------------------------
def nli_build_batches(premises, entity_list, max_len, tok, device, bs=16):
    """
    For each sample, build two NLI pairs:
      (premise, H_yes), (premise, H_no)
    Returns iterable of encoded batches, plus index mapping to collect scores.
    """
    H_YES_TMPL = "The mention of {} conveys an opinion or sentiment."
    H_NO_TMPL  = "The mention of {} is purely objective and contains no opinion."

    inputs = []
    for premise, ent in zip(premises, entity_list):
        h_yes = H_YES_TMPL.format(ent if ent else "the entity")
        h_no  = H_NO_TMPL.format(ent if ent else "the entity")
        inputs.append((premise, h_yes, h_no))

    # batching
    for i in range(0, len(inputs), bs):
        batch = inputs[i:i+bs]
        prem  = [b[0] for b in batch]
        h_yes = [b[1] for b in batch]
        h_no  = [b[2] for b in batch]

        enc_yes = tok(prem, h_yes, padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt").to(device)
        enc_no  = tok(prem, h_no,  padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt").to(device)
        yield i, enc_yes, enc_no, len(batch)

def run_nli_binary_emotion(df, device, nli_cache,
                           max_len=256, bs=16, bias=0.7, use_bias=True):
    """
    Output: Nx2 tensor probs [p(no-emotion), p(emotion)]
    We compute entailment logits for (H_yes, H_no) and apply optional label-aware bias.
    """
    tok = AutoTokenizer.from_pretrained(nli_cache, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(nli_cache).to(device).eval()

    # premise = filled Description + ctx_text
    filled_desc = [replace_T(d, e) for d, e in zip(df["text"].tolist(), df["entity"].tolist())]
    premises = [(fd + " " + cx).strip() if cx else fd for fd, cx in zip(filled_desc, df["ctx_text"].tolist())]

    # map original labels -> coarse: 1->0, 0/2->1
    orig = df["label"].tolist()
    coarse = [0 if x == 1 else 1 for x in orig]  # 0=no-emotion, 1=emotion

    # storage
    N = len(df)
    out = torch.zeros((N, 2), dtype=torch.float32)

    with torch.inference_mode():
        for i, enc_yes, enc_no, n in tqdm(nli_build_batches(premises, df["entity"].tolist(),
                                                            max_len, tok, device, bs),
                                          total=math.ceil(N/bs), desc="NLI (emotion presence)"):
            # logits shape: (n, 3) -> [contradiction, neutral, entailment] usually
            logit_yes = mdl(**enc_yes).logits  # (n,3)
            logit_no  = mdl(**enc_no).logits   # (n,3)

            # pick entailment logit
            # safer via id2label:
            id2label = mdl.config.id2label
            ent_id = None
            for k, v in id2label.items():
                if str(v).upper().startswith("ENTAIL"):  # ENTAILMENT
                    ent_id = int(k)
                    break
            if ent_id is None:
                # fallback to last index
                ent_id = 2

            s_yes = logit_yes[:, ent_id]  # (n,)
            s_no  = logit_no[:,  ent_id]  # (n,)

            # label-aware bias (supervised calibration)
            if use_bias and bias != 0.0:
                # add bias to the ground-truth coarse class logit
                # coarse: 0 -> favor no-emotion (s_no), 1 -> favor emotion (s_yes)
                batch_coarse = torch.tensor(coarse[i:i+n], device=s_yes.device)
                s_yes = s_yes + (batch_coarse == 1).float() * bias
                s_no  = s_no  + (batch_coarse == 0).float() * bias

            # binary probs via softmax over [no, yes]
            logits_bin = torch.stack([s_no, s_yes], dim=1)  # (n,2)
            probs = softmax(logits_bin, dim=1)  # (n,2): [p(no), p(yes)]

            out[i:i+n] = probs.cpu()

    return out

# -------------------------
# Teacher 2: CLIP ITM -> Match (binary)
# -------------------------
def load_images_batch(paths):
    imgs = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            img = None
        imgs.append(img)
    return imgs

def run_clip_match(df, device, clip_cache,
                   img_root="", bs=32, tau=1.0):
    """
    Output: Nx2 tensor probs [p(unmatch), p(match)]
    Text prompt fuses Description (with $T->Entity) + explicit entity.
    Temperature tau controls calibration (lower -> sharper).
    """
    proc = CLIPProcessor.from_pretrained(clip_cache)
    mdl  = CLIPModel.from_pretrained(clip_cache).to(device).eval()

    # build image paths
    img_paths = []
    for p in df["image_path"].tolist():
        p = str(p)
        if img_root and not os.path.isabs(p):
            img_paths.append(os.path.join(img_root, p))
        else:
            img_paths.append(p)

    # build text prompts
    desc_filled = [replace_T(d, e) for d, e in zip(df["text"].tolist(), df["entity"].tolist())]
    prompts = [f"{d}. The entity is: {e}." if e else d for d, e in zip(desc_filled, df["entity"].tolist())]

    N = len(df)
    out = torch.zeros((N, 2), dtype=torch.float32)

    with torch.inference_mode():
        for i in tqdm(range(0, N, bs), desc="CLIP ITM (match)"):
            sub_paths = img_paths[i:i+bs]
            sub_prompts = prompts[i:i+bs]
            imgs = load_images_batch(sub_paths)

            # For missing images or empty text -> neutral 0.5/0.5
            mask_bad = [ (im is None) or (not (t and t.strip())) for im, t in zip(imgs, sub_prompts) ]
            if all(mask_bad):
                out[i:i+bs] = torch.tensor([[0.5, 0.5]] * len(sub_prompts))
                continue

            # Replace bad samples with a 1x1 blank image to keep batch shape; we will overwrite later
            safe_imgs = [im if im is not None else Image.new("RGB", (1,1), (128,128,128)) for im in imgs]
            inputs = proc(text=sub_prompts, images=safe_imgs, return_tensors="pt", padding=True).to(device)

            outputs = mdl(**inputs)
            # cosine-like logits per image-text
            # shape: (B, B) but processor paired them; we want the diagonal
            # (CLIPProcessor pairs images/text 1-to-1)
            sim = outputs.logits_per_image  # (B,B) or (B,1) depending on processor pairing
            if sim.dim() == 2:
                s = torch.diagonal(sim, 0)  # (B,)
            else:
                s = sim.view(-1)            # (B,)

            # binary logits [unmatch, match] using a temperature tau
            # baseline used [s, -s], here: [ -s/tau, +s/tau ]
            logits_bin = torch.stack([ -s / tau, s / tau ], dim=1)  # (B,2)
            probs = softmax(logits_bin, dim=1).cpu()  # [p(unmatch), p(match)]

            # bad ones -> 0.5
            for j, bad in enumerate(mask_bad):
                if bad:
                    probs[j] = torch.tensor([0.5, 0.5])

            out[i:i+bs] = probs

    return out

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="CSV/TSV/JSON/JSONL")
    ap.add_argument("--out_path", required=True, help="CSV or JSON(L); inferred by extension")
    ap.add_argument("--img_root", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_dir",default=r"models", required=True, help="Local cache directory, e.g., ./models")
    ap.add_argument("--nli_model", default="roberta-large-mnli")
    ap.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--sep", default="auto")

    # New knobs
    ap.add_argument("--nli_bs", type=int, default=16)
    ap.add_argument("--clip_bs", type=int, default=32)
    ap.add_argument("--nli_bias", type=float, default=0.7, help="Label-aware bias added to correct class logit")
    ap.add_argument("--nli_use_bias", type=int, default=1, help="1 to enable supervised bias; 0 to disable")
    ap.add_argument("--clip_tau", type=float, default=1.0, help="Temperature for CLIP binary calibration (lower=sharper)")

    args = ap.parse_args()

    # IO
    df = load_any(args.in_path, sep=args.sep)
    df = normalize_df(df)

    # Prime caches
    nli_cache = ensure_cache(args.nli_model, os.path.join(args.model_dir, args.nli_model.replace("/", "-")))
    clip_cache = ensure_cache(args.clip_model, os.path.join(args.model_dir, args.clip_model.replace("/", "-")))

    # Teacher 1: NLI (emotion presence wrt Entity)
    probs_tf = run_nli_binary_emotion(
        df, device=args.device, nli_cache = nli_cache,
        max_len=args.max_len, bs=args.nli_bs, bias=args.nli_bias, use_bias=bool(args.nli_use_bias)
    )  # (N,2) => [p(no-emotion), p(emotion)]

    # Teacher 2: CLIP (match)
    probs_it = run_clip_match(
        df, device=args.device, clip_cache=clip_cache,
        img_root=args.img_root, bs=args.clip_bs, tau=args.clip_tau
    )  # (N,2) => [p(unmatch), p(match)]

    # Write back as JSON strings
    df["teacher_tf"] = [to_json_list(row.tolist()) for row in probs_tf]
    df["teacher_it"] = [to_json_list(row.tolist()) for row in probs_it]

    #  coarse_label：0=no-emotion, 1=emotion
    df["coarse_label"] = [0 if x == 1 else 1 for x in df["label"].tolist()]

    save_any(df, args.out_path)
    print(f"Saved -> {args.out_path} ({len(df)} rows).")

if __name__ == "__main__":
    main()
