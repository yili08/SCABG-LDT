# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
import math
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

from dataset_v2 import FNDatasetV2
from multimodal_mh_upgraded import SCABG_LDT_Model, UpgConfig


def _load_labels_for_sampler(train_csv_path):
    import pandas as pd
    ext = os.path.splitext(train_csv_path)[1].lower()
    if ext in ['.json', '.jsonl', '.jsn']:
        try:
            df = pd.read_json(train_csv_path, lines=True)
        except Exception:
            df = pd.read_json(train_csv_path)
    else:
        try:
            df = pd.read_csv(train_csv_path, sep=None, engine='python', encoding='utf-8-sig')
        except Exception:
            df = pd.read_csv(train_csv_path, encoding='utf-8-sig')
    cols = {c.lower(): c for c in df.columns}
    key = cols.get('label', cols.get('Label', None))
    if key is None:
        raise ValueError('not found label / Label ')
    return df[key].astype(int).tolist()


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = set().union(*[b.keys() for b in batch])
    for k in keys:
        vals = [b[k] for b in batch if k in b]
        v0 = vals[0]
        if torch.is_tensor(v0):
            try: out[k] = torch.stack(vals, dim=0)
            except Exception: out[k] = vals
        else:
            out[k] = vals
    return out


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             dataloader: DataLoader,
             device: str,
             num_classes: int = 3,
             verbose: bool = True,
             logit_adjust: Optional[torch.Tensor] = None) -> Dict[str, float]:
    
    model.eval()
    all_preds, all_labels = [], []
    for batch in dataloader:
        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        out = model(
            image=batch["image"],
            news_ids=batch["news_ids"],
            news_mask=batch["news_mask"],
            ctx_ids=batch.get("ctx_ids"),
            ctx_mask=batch.get("ctx_mask"),
        )
        logits = out["logits"]
        if logit_adjust is not None:
            logits = logits + logit_adjust  
        preds = logits.argmax(dim=-1)
        labels = batch["labels"]
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", labels=list(range(num_classes)))
    if verbose:
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
        print("[Eval] acc={:.4f}  f1={:.4f}".format(acc, macro_f1))
        print("[Eval] Confusion:\n", cm)
        print(classification_report(all_labels, all_preds, labels=list(range(num_classes)), zero_division=0))
    return {"acc": float(acc), "f1": float(macro_f1)}


def build_scheduler(optimizer,
                    scheduler_name: str,
                    total_steps: int,
                    base_lr: float,
                    warmup_ratio: float = 0.05,
                    eta_min_ratio: float = 0.1):
    
    scheduler_name = (scheduler_name or "none").lower()
    warmup_steps = int(total_steps * max(0.0, warmup_ratio))
    warmup_steps = min(warmup_steps, max(1, total_steps - 1))

    if scheduler_name == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return max(step / float(max(1, warmup_steps)), 1e-8)
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_min_ratio + (1.0 - eta_min_ratio) * cosine
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    elif scheduler_name == "linear":
        def lr_lambda(step):
            if step < warmup_steps:
                return max(step / float(max(1, warmup_steps)), 1e-8)
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 1.0 - progress)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    else:
        return None


class ModelEMA:
        def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

        self.backup = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert n in self.shadow
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=(1.0 - self.decay))

    def store(self, model: torch.nn.Module):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].data)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        if not self.backup:
            return
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n].data)
        self.backup = {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True, type=str)
    parser.add_argument("--val_csv", required=True, type=str)
    parser.add_argument("--img_root", default="", type=str)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=160)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="ckpt_upg.pt")
    parser.add_argument("--use_ctx", action="store_true")

    # image & norm
    parser.add_argument("--img_encoder", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--img_norm", type=str, default="vit", choices=["vit", "imagenet"])

    # imbalance
    parser.add_argument('--oversample_neg_factor', type=float, default=2.0,
                        help='>1.0 for label==0')

    parser.add_argument('--init_from', type=str, default='')

    # loss weights & temps (lower KD)
    parser.add_argument("--lambda_kd", type=float, default=0.02)
    parser.add_argument("--lambda_logic", type=float, default=0.0)
    parser.add_argument("--lambda_coarse", type=float, default=0.3)
    parser.add_argument("--lambda_fine", type=float, default=0.3)
    parser.add_argument("--lambda_adv", type=float, default=0.001)
    parser.add_argument("--lambda_contrast", type=float, default=0.005)
    parser.add_argument("--lambda_band", type=float, default=0.005)
    parser.add_argument("--temp", type=float, default=3.0)
    parser.add_argument("--contrastive_temp", type=float, default=0.07)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    # warmups
    parser.add_argument("--adv_warmup_steps", type=int, default=12000)
    parser.add_argument("--kd_warmup_steps", type=int, default=6000)
    parser.add_argument("--contrast_warmup_steps", type=int, default=9000)
    parser.add_argument("--band_warmup_steps", type=int, default=9000)
    parser.add_argument("--grl_max_lambd", type=float, default=0.05)

    # focal
    parser.add_argument("--use_focal", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    # scheduler (+warmup)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "linear", "none"])
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="lr / warmup ")
    parser.add_argument("--eta_min_ratio", type=float, default=0.1, help="cosine  lr ")

    # dynamic CF weights
    parser.add_argument("--external_dynamic_cf", action="store_true",
                        help="out EMA updata λ_coarse/λ_fine")
    parser.add_argument("--external_cf_ema", type=float, default=0.9)

    # Late unreg & KD cap
    parser.add_argument("--late_start", type=float, default=0.6, help="training progress exceeds this percentage")
    parser.add_argument("--late_ctr_decay", type=float, default=0.25)
    parser.add_argument("--late_adv_decay", type=float, default=0.5)
    parser.add_argument("--late_kd_cap", type=float, default=0.20)

    # EMA
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)

    # Logit Adjustment 
    parser.add_argument("--logit_adjust_tau", type=float, default=1.0)

    # Class-Balanced (Effective Number)
    parser.add_argument("--cb_beta", type=float, default=0.999)

    # AMP
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # datasets
    train_ds = FNDatasetV2(
        csv_path=args.train_csv,
        image_root=args.img_root,
        max_len=args.max_len,
        use_ctx=args.use_ctx,
        train=True,
        img_size=args.img_size,
        img_norm=("vit" if args.img_encoder.lower().startswith("google/vit") or args.img_encoder.lower().startswith("vit") else args.img_norm)
    )
    val_ds = FNDatasetV2(
        csv_path=args.val_csv,
        image_root=args.img_root,
        max_len=args.max_len,
        use_ctx=args.use_ctx,
        train=True,
        img_size=args.img_size,
        img_norm=("vit" if args.img_encoder.lower().startswith("google/vit") or args.img_encoder.lower().startswith("vit") else args.img_norm)
    )

    # histogram & priors
    labels_np = train_ds.df["label"].astype(int).values
    num_classes = 3
    hist = np.bincount(labels_np, minlength=num_classes).astype(float)
    freq = hist / (hist.sum() + 1e-8)

    # ---- Class-Balanced weights (Effective Number) ----
    beta = float(args.cb_beta)
    eff_num = 1.0 - np.power(beta, hist + 1e-12)
    cb_weights = (1.0 - beta) / np.clip(eff_num, 1e-12, None)
    cb_weights = cb_weights / (cb_weights.mean() + 1e-12)  
    class_weights = cb_weights  

    # ---- Logit Adjustment  ----
    prior = freq  # p(y)
    logit_adjust_np = args.logit_adjust_tau * np.log(np.clip(prior, 1e-12, None))
    logit_adjust = torch.tensor(logit_adjust_np, dtype=torch.float32, device=device)

    # oversample negatives
    train_sampler = None
    if args.oversample_neg_factor and args.oversample_neg_factor > 1.0:
        labels_in_file = _load_labels_for_sampler(args.train_csv)
        if len(labels_in_file) != len(train_ds):
            labels_in_file = labels_in_file[:len(train_ds)]
        weights = [args.oversample_neg_factor if int(y) == 0 else 1.0 for y in labels_in_file]
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True
        )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
        drop_last=False
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
        drop_last=False
    )

    # model & cfg
    cfg = UpgConfig(
        use_ctx=args.use_ctx,
        num_classes=num_classes,
        lambda_kd=args.lambda_kd,
        lambda_logic=args.lambda_logic,
        lambda_adv=args.lambda_adv,
        temp=args.temp,
        lambda_coarse=args.lambda_coarse,
        lambda_fine=args.lambda_fine,
        lambda_contrast=args.lambda_contrast,
        lambda_band=args.lambda_band,
        contrastive_temp=args.contrastive_temp,
        ce_label_smoothing=args.label_smoothing,
        class_weights=class_weights.tolist(),   
        img_encoder=args.img_encoder,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        gate_self_attn=True,
        dynamic_cf_weights=True  
    )

    model = SCABG_LDT_Model(cfg).to(device)

    if args.init_from and os.path.isfile(args.init_from):
        ckpt = torch.load(args.init_from, map_location='cpu')
        sd = ckpt.get('model', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[InitFrom] {args.init_from} | missing={len(missing)} unexpected={len(unexpected)}")

    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    def build_param_groups(m, base_lr, head_mult=20.0):
        head_words = ['classifier', 'cls', 'coarse', 'fine', 'fusion', 'gate', 'detector.fc', 'head']
        enc, heads = [], []
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            if any(hw in n.lower() for hw in head_words):
                heads.append(p)
            else:
                enc.append(p)
        return [
            {'params': enc, 'lr': base_lr},
            {'params': heads, 'lr': base_lr * head_mult}
        ]

    optimizer = torch.optim.AdamW(
        build_param_groups(model, args.lr, head_mult=20.0),
        weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    total_steps = args.epochs * max(1, len(train_dl))
    scheduler = build_scheduler(
        optimizer, args.scheduler, total_steps, args.lr,
        warmup_ratio=args.warmup_ratio, eta_min_ratio=args.eta_min_ratio
    )

    # EMA
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    def lin_warm(curr_step: int, total_warm_steps: int) -> float:
        if total_warm_steps <= 0: return 1.0
        return min(1.0, curr_step / float(total_warm_steps))

    global_step = 0
    best_f1 = -1.0
    best_path = args.save

    #  dynamic coarse/fine 
    ext_cf_use = args.external_dynamic_cf
    cf_m = float(args.external_cf_ema)
    coarse_ema, fine_ema = None, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        for it, batch in enumerate(train_dl, start=1):
            global_step += 1

            
            adv_scale = lin_warm(global_step, args.adv_warmup_steps)
            kd_scale  = lin_warm(global_step, args.kd_warmup_steps)
            ctr_scale = lin_warm(global_step, args.contrast_warmup_steps)
            band_scale= lin_warm(global_step, args.band_warmup_steps)
            grl_lambda = args.grl_max_lambd * adv_scale

            
            progress = global_step / float(max(1, total_steps))
            late = (progress > args.late_start)
            if late:
                ctr_scale *= args.late_ctr_decay
                adv_scale *= args.late_adv_decay
                kd_scale = min(kd_scale, args.late_kd_cap / max(1e-12, args.lambda_kd)) if args.lambda_kd > 0 else kd_scale

            
            model.cfg.lambda_adv_curr = args.lambda_adv * adv_scale
            model.cfg.grl_lambd_curr  = grl_lambda
            model.cfg.lambda_kd_curr  = args.lambda_kd * kd_scale
            model.cfg.lambda_contrast_curr = args.lambda_contrast * ctr_scale
            model.cfg.lambda_band_curr     = args.lambda_band * band_scale
            # coarse/fine
            model.cfg.lambda_coarse_curr   = args.lambda_coarse
            model.cfg.lambda_fine_curr     = args.lambda_fine

            for k, v in list(batch.items()):
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)

            if "coarse" not in batch or batch["coarse"] is None:
                if "labels" in batch:
                    batch["coarse"] = (batch["labels"] != 1).long()

            def _ensure_fp_tensor(x):
                if x is None: return None
                if torch.is_tensor(x): return x.float()
                return torch.tensor(x, dtype=torch.float32, device=device)

            teacher_tf = _ensure_fp_tensor(batch.get("teacher_tf"))
            teacher_in = _ensure_fp_tensor(batch.get("teacher_in"))
            teacher_it = _ensure_fp_tensor(batch.get("teacher_it"))

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                out = model(
                    image=batch["image"],
                    news_ids=batch["news_ids"],
                    news_mask=batch["news_mask"],
                    ctx_ids=batch.get("ctx_ids"),
                    ctx_mask=batch.get("ctx_mask"),
                    labels=batch.get("labels"),
                    teacher_tf=teacher_tf,
                    teacher_in=teacher_in,
                    teacher_it=teacher_it,
                    coarse=batch.get("coarse"),
                )

                
                if ext_cf_use:
                    c_loss = float(out.get("coarse_loss", 0.0))
                    f_loss = float(out.get("fine_loss", 0.0))
                    if coarse_ema is None:
                        coarse_ema, fine_ema = c_loss, f_loss
                    else:
                        coarse_ema = cf_m * coarse_ema + (1 - cf_m) * c_loss
                        fine_ema   = cf_m * fine_ema   + (1 - cf_m) * f_loss
                    inv_c = 1.0 / (coarse_ema + 1e-6)
                    inv_f = 1.0 / (fine_ema   + 1e-6)
                    w_c = inv_c / (inv_c + inv_f)
                    w_f = 1.0 - w_c
                    total_cf = max(1e-12, args.lambda_coarse + args.lambda_fine)
                    model.cfg.lambda_coarse_curr = total_cf * w_c
                    model.cfg.lambda_fine_curr   = total_cf * w_f

                loss = out["loss"]

            if loss is None:
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)

            if (it % 20) == 0:
                log_keys = [
                    "loss", "ce", "kd_loss", "coarse_loss", "fine_loss",
                    "logic_loss", "adv_loss", "contrastive_loss", "band_consistency",
                    "dyn_coarse_w", "dyn_fine_w"
                ]
                msg = [f"[Train] ep {epoch} it {it}"]
                for k in log_keys:
                    v = out.get(k, None)
                    if isinstance(v, torch.Tensor):
                        v = float(v.detach().cpu().item())
                    msg.append(f"{k}:{v:.4f}" if isinstance(v, float) else f"{k}:{v}")
                curr_lr = optimizer.param_groups[0]['lr']
                msg.append(f"lr:{curr_lr:.2e}")
                msg.append(f"adv_scale:{adv_scale:.2f} grl:{grl_lambda:.2f} kd_scale:{kd_scale:.2f} ctr_scale:{ctr_scale:.2f}")
                if ext_cf_use:
                    msg.append(f"ext_c:{getattr(model.cfg,'lambda_coarse_curr',0):.4f} ext_f:{getattr(model.cfg,'lambda_fine_curr',0):.4f}")
                if late:
                    msg.append("late:1")
                print(" | ".join(msg))

        # === ===
        if ema is not None:
            ema.store(model); ema.copy_to(model)
        metrics = evaluate(model, val_dl, device, num_classes=num_classes, verbose=True, logit_adjust=logit_adjust)
        if ema is not None:
            ema.restore(model)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_obj = {"model": (ema.shadow if ema is not None else model.state_dict()), "metrics": metrics}
            
            torch.save(save_obj, args.save)
            print(f"[Save] best f1={best_f1:.4f} -> {args.save}")

    print(f"Best Macro-F1: {best_f1:.4f} | ckpt: {args.save}")


if __name__ == "__main__":
    main()
