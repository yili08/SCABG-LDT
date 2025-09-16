# -*- coding: utf-8 -*-
"""
Upgraded multimodal model with:
- SCABG branches
- ViT image encoder (default google/vit-base-patch16-224) or ResNet34 fallback
- Attention-based BandAggregator
- (NEW) Cross-branch Multi-Head Attention fusion before classifier
- (NEW) Dynamic weights for coarse/fine losses via homoscedastic uncertainty
- Optional Focal Loss
- KD (lower-weight, safe-dimension check) + delayed/adaptive GRL adversarial heads
"""
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.hub.set_dir("models")

# -----------------------------
# Helpers
# -----------------------------
def l2norm(x, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    a = l2norm(a, -1); b = l2norm(b, -1)
    logits = a @ b.t() / temperature
    labels = torch.arange(a.size(0), device=a.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) * 0.5
    return loss

def focal_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor,
                           alpha: Optional[torch.Tensor] = None, gamma: float = 2.0) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    idx = torch.arange(probs.size(0), device=probs.device)
    pt = probs[idx, targets]
    log_pt = log_probs[idx, targets]
    if alpha is not None:
        at = alpha[targets]
        log_pt = log_pt * at
    loss = - ((1.0 - pt) ** gamma) * log_pt
    return loss.mean()

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

class GRL(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return GradReverse.apply(x, self.lambd)

# -----------------------------
# Simple 1D DFT (text freq features optional)
# -----------------------------
def dft_1d(seq: torch.Tensor) -> torch.Tensor:
    orig_dtype = seq.dtype
    with torch.cuda.amp.autocast(enabled=False):
        f = torch.fft.rfft(seq.float(), dim=1, norm="ortho")
        real = F.interpolate(f.real.transpose(1, 2), size=seq.shape[1],
                             mode="linear", align_corners=False).transpose(1, 2)
        imag = F.interpolate(f.imag.transpose(1, 2), size=seq.shape[1],
                             mode="linear", align_corners=False).transpose(1, 2)
        out = torch.cat([real, imag], dim=-1)
    return out.to(orig_dtype)

# -----------------------------
# Blocks
# -----------------------------
class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(d_hidden, d_out),
        )
    def forward(self, x):
        return self.net(x)

class LatentCrossAttention(nn.Module):
    
    def __init__(self, hidden: int, num_queries: int = 8, heads: int = 8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(num_queries, hidden) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=heads, batch_first=True)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.size(0)
        Q = self.q.unsqueeze(0).repeat(B, 1, 1)
        out, _ = self.attn(Q, tokens, tokens)
        out = self.out(out.mean(dim=1))
        return out

class ComplexBandGate(nn.Module):
    
    def __init__(self, hidden: int, K: int = 3, cond_dim: int = None, use_self_attn: bool = True, nheads: int = 4):
        super().__init__()
        self.K = K
        self.use_self_attn = use_self_attn
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=nheads, batch_first=True) if use_self_attn else None
        self.ln = nn.LayerNorm(hidden) if use_self_attn else None

        self.proj_tok = nn.Linear(hidden, hidden)
        self.proj_cond = nn.Linear(cond_dim, hidden) if cond_dim is not None else None
        self.gate = nn.Linear(hidden, K)

    def forward(self, tokens: torch.Tensor, cond: Optional[torch.Tensor] = None):
        h = tokens
        if self.use_self_attn and self.self_attn is not None:
            h2, _ = self.self_attn(h, h, h)
            h = self.ln(h + h2)
        h = self.proj_tok(h)
        if cond is not None and self.proj_cond is not None:
            c = self.proj_cond(cond).unsqueeze(1)
            h = h + c
        scores = self.gate(torch.tanh(h))     # (B,N,K)
        probs = torch.softmax(scores, dim=-1) # (B,N,K)
        bands = []
        for k in range(self.K):
            bands.append(tokens * probs[..., k:k+1])
        return bands, probs.mean(dim=1)       # (list of K (B,N,H), (B,K))

class BandAggregator(nn.Module):
    
    def __init__(self, hidden: int, K: int = 3):
        super().__init__()
        self.fuse = nn.ModuleList([MLP(2*hidden, hidden, hidden) for _ in range(K)])

    def forward(self, hx_list: List[torch.Tensor], hy_list: List[torch.Tensor], cond: torch.Tensor):
        fused_per_band = []
        for k, (hx, hy) in enumerate(zip(hx_list, hy_list)):
            fused = self.fuse[k](torch.cat([hx, hy], dim=-1))
            fused_per_band.append(fused)
        stacked = torch.stack(fused_per_band, dim=1)            # (B,K,H)
        scores = torch.bmm(stacked, cond.unsqueeze(2)).squeeze(2)  # (B,K)
        gate_w = torch.softmax(scores, dim=-1)                  # (B,K)
        weighted = (gate_w.unsqueeze(-1) * stacked).sum(dim=1)  # (B,H)
        return weighted, gate_w, fused_per_band

# -----------------------------
# Encoders
# -----------------------------
class ImageEncoder(nn.Module):
    def __init__(self, out_dim=512, encoder_type: str = "google/vit-base-patch16-224"):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        if self.encoder_type.startswith("google/vit"):
            from transformers import ViTModel
            self.vit = ViTModel.from_pretrained(encoder_type, cache_dir="models")
            self.proj = nn.Linear(self.vit.config.hidden_size, out_dim)
            self.is_vit = True
        elif self.encoder_type.startswith("vit"):
            
            from transformers import ViTModel
            self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224", cache_dir="models")
            self.proj = nn.Linear(self.vit.config.hidden_size, out_dim)
            self.is_vit = True
        else:
            from torchvision.models import resnet34, ResNet34_Weights
            weights = ResNet34_Weights.DEFAULT
            base = resnet34(weights=weights)
            self.backbone = nn.Sequential(*list(base.children())[:-2])  # (B,512,H/32,W/32)
            self.proj_conv = nn.Conv2d(512, out_dim, 1)
            self.is_vit = False

    def forward(self, x):
        if self.is_vit:
            out = self.vit(x)
            tok = out.last_hidden_state[:, 1:, :]  
            tok = self.proj(tok)
            return tok                                # (B, N, out_dim)
        else:
            f = self.backbone(x)
            f = self.proj_conv(f)                    # (B,out_dim,H/32,W/32)
            return f

class TextEncoder(nn.Module):
    def __init__(self, model_name="roberta-large", out_dim=768, pretrained=True):
        super().__init__()
        from transformers import AutoModel, AutoConfig
        cfg = AutoConfig.from_pretrained(model_name, cache_dir="models")
        self.transformer = AutoModel.from_pretrained(model_name, cache_dir="models") if pretrained \
            else AutoModel.from_config(cfg)
        self.out_dim = out_dim
    def forward(self, input_ids, attention_mask):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        cls = last[:, 0]
        return last, cls

# -----------------------------
# SCABG Branch
# -----------------------------
class SCABGBranch(nn.Module):
    def __init__(self, hidden: int, K: int = 3, num_queries: int = 8, heads: int = 8, p_dim: int = 128, name: str = "branch", gate_self_attn: bool = True):
        super().__init__()
        self.name = name
        self.K = K
        self.gate_x = ComplexBandGate(hidden, K, cond_dim=hidden, use_self_attn=gate_self_attn)
        self.gate_y = ComplexBandGate(hidden, K, cond_dim=hidden, use_self_attn=gate_self_attn)
        self.lca_x = nn.ModuleList([LatentCrossAttention(hidden, num_queries, heads) for _ in range(K)])
        self.lca_y = nn.ModuleList([LatentCrossAttention(hidden, num_queries, heads) for _ in range(K)])
        self.agg   = BandAggregator(hidden, K)
        self.head  = MLP(hidden, hidden, p_dim)

    def forward(self, tokens_x: torch.Tensor, tokens_y: torch.Tensor, cond: torch.Tensor, return_band_info=False):
        bands_x, dist_x = self.gate_x(tokens_x, cond)
        bands_y, dist_y = self.gate_y(tokens_y, cond)
        hx_list, hy_list = [], []
        for k in range(self.K):
            hx_list.append(self.lca_x[k](bands_x[k]))
            hy_list.append(self.lca_y[k](bands_y[k]))
        fused, gate_w, fused_per_band = self.agg(hx_list, hy_list, cond)
        p = self.head(fused)

        aux = None
        if return_band_info:
            aux = {
                "band_gate": gate_w,
                "band_x": dist_x,
                "band_y": dist_y,
                "hx_list": hx_list, "hy_list": hy_list,
                "fused_per_band": fused_per_band
            }
        return p, aux

# -----------------------------
# Detection & Distillation Head
# -----------------------------
class DetectionHead(nn.Module):
    
    
    def __init__(self, p_dim=128, num_classes=3, kd_dims=(2,0,2), entail_indices=(0,0,0), use_dynamic_cf=True):
        super().__init__()
        self.num_classes = num_classes
        self.net_it = MLP(p_dim, 256, 128)
        self.net_tf = MLP(p_dim, 256, 128)
        self.net_in = MLP(p_dim, 256, 128)

        # NEW: cross-branch attention (3 tokens: it/tf/in)
        self.branch_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.branch_ln = nn.LayerNorm(128)
        self.fuse = MLP(128, 256, 128)

        self.coarse_cls = nn.Linear(128, 2)
        self.fine_cls   = nn.Linear(128, 2)

        # KD heads
        kd_tf_dim, kd_in_dim, kd_it_dim = kd_dims
        self.kd_tf = nn.Linear(128, kd_tf_dim) if kd_tf_dim > 0 else None
        self.kd_in = nn.Linear(128, kd_in_dim) if kd_in_dim > 0 else None
        self.kd_it = nn.Linear(128, kd_it_dim) if kd_it_dim > 0 else None
        self.entail_idx_tf, self.entail_idx_in, self.entail_idx_it = entail_indices

        # adversarial
        self.grl = GRL(lambd=0.0)
        self.adv_it = nn.Linear(128, num_classes)
        self.adv_tf = nn.Linear(128, num_classes)
        self.adv_in = nn.Linear(128, num_classes)

        # dynamic weights (homoscedastic uncertainty)
        self.use_dynamic_cf = use_dynamic_cf
        if self.use_dynamic_cf:
            self.log_var_coarse = nn.Parameter(torch.zeros(1))
            self.log_var_fine   = nn.Parameter(torch.zeros(1))

        self.cfg = None

        self.mix_lambda = nn.Parameter(torch.tensor(0.7))  
        self.pool_w = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))  

    def logic_score(self, logit_tf, logit_in, logit_it):
        def prob_entail(logits, idx):
            if logits is None: return None
            p = torch.softmax(logits, dim=-1); return p[:, idx]
        comps = [x for x in [prob_entail(logit_tf, self.entail_idx_tf),
                             prob_entail(logit_in, self.entail_idx_in),
                             prob_entail(logit_it, self.entail_idx_it)] if x is not None]
        if not comps: return None
        s = comps[0]
        for c in comps[1:]: s = s * c
        return s

    def forward(self, p_it, p_tf, p_in,
                labels: Optional[torch.Tensor] = None,
                coarse_labels: Optional[torch.Tensor] = None,
                kd_teachers: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:

        d_it = self.net_it(p_it)
        d_tf = self.net_tf(p_tf)
        d_in = self.net_in(p_in)

        # NEW: Cross-branch attention fusion
        lam = self.mix_lambda.sigmoid()  
        tokens = torch.stack([d_tf, lam * d_tf + (1 - lam) * d_in, d_it], dim=1)  # (B,3,D)
        attn_out, _ = self.branch_attn(tokens, tokens, tokens)
        tokens = self.branch_ln(tokens + attn_out)          # (B,3,128)
        # fused = tokens.mean(dim=1)                          # (B,128)
        # fused = (0.45 * tokens[:, 0] + 0.35 * tokens[:, 1] + 0.20 * tokens[:, 2])
        # fused = torch.einsum('bnd,n->bn', tokens, self.pool_w.softmax(dim=0))
        w = torch.softmax(self.pool_w, dim=0)  # [N]
        fused = (tokens * w.view(1, -1, 1)).sum(dim=1)  # [B, D]
        fused = self.fuse(fused)

        coarse_logits = self.coarse_cls(fused)              # (B,2)
        fine_logits   = self.fine_cls(fused)                # (B,2)

        coarse_logit_neu = coarse_logits[:, 0]
        coarse_logit_emo = coarse_logits[:, 1]
        final_logit_neg  = coarse_logit_emo + fine_logits[:, 0]
        final_logit_pos  = coarse_logit_emo + fine_logits[:, 1]
        final_logit_neu  = coarse_logit_neu
        final_logits = torch.stack([final_logit_neg, final_logit_neu, final_logit_pos], dim=1)

        out: Dict[str, torch.Tensor] = {
            "logits": final_logits,
            "kd_tf_logits": self.kd_tf(d_tf) if self.kd_tf is not None else None,
            "kd_in_logits": self.kd_in(d_in) if self.kd_in is not None else None,
            "kd_it_logits": self.kd_it(d_it) if self.kd_it is not None else None,
        }

        total_loss = None
        ce_loss = kd_loss = coarse_loss = fine_loss = logic_loss = adv_loss = None
        dyn_cw = dyn_fw = None

        if labels is not None:
            ce_weight  = None
            smoothing  = 0.0
            use_focal  = False
            focal_gamma = 2.0
            lambda_coarse = 0.0
            lambda_fine   = 0.0
            lambda_kd     = 0.0
            lambda_logic  = 0.0
            lambda_adv    = 0.0
            if hasattr(self, "cfg") and self.cfg is not None:
                if getattr(self.cfg, "class_weights", None) is not None:
                    w = torch.tensor(self.cfg.class_weights, device=final_logits.device, dtype=final_logits.dtype)
                    if w.numel() == final_logits.size(-1): ce_weight = w
                smoothing = getattr(self.cfg, "ce_label_smoothing", 0.0)
                use_focal = getattr(self.cfg, "use_focal", False)
                focal_gamma = getattr(self.cfg, "focal_gamma", 2.0)
                lambda_coarse = getattr(self.cfg, "lambda_coarse_curr", getattr(self.cfg, "lambda_coarse", 0.3))
                lambda_fine   = getattr(self.cfg, "lambda_fine_curr",   getattr(self.cfg, "lambda_fine",   0.3))
                lambda_kd     = getattr(self.cfg, "lambda_kd_curr",     getattr(self.cfg, "lambda_kd",     0.03))
                lambda_logic  = getattr(self.cfg, "lambda_logic_curr",  getattr(self.cfg, "lambda_logic",  0.0))
                lambda_adv    = getattr(self.cfg, "lambda_adv_curr",    getattr(self.cfg, "lambda_adv",    0.001))
                if hasattr(self.cfg, "grl_lambd_curr"):
                    self.grl.lambd = self.cfg.grl_lambd_curr

            # CE / Focal
            if use_focal:
                alpha = ce_weight if ce_weight is not None else None
                ce_loss = focal_loss_from_logits(final_logits, labels, alpha=alpha, gamma=focal_gamma)
            else:
                ce_loss = F.cross_entropy(final_logits, labels, weight=ce_weight, label_smoothing=smoothing)
            total_loss = ce_loss

            # Coarse/Fine
            if coarse_labels is None:
                coarse_labels = (labels != 1).long()
            coarse_loss = F.cross_entropy(coarse_logits, coarse_labels, reduction='mean')

            fine_targets = (labels == 2).long()
            mask = (coarse_labels == 1)
            if mask.any():
                fine_loss = F.cross_entropy(fine_logits[mask], fine_targets[mask], reduction='mean')
            else:
                fine_loss = torch.tensor(0.0, device=final_logits.device)

            # (NEW) dynamic weights
            if self.use_dynamic_cf:
                # Kendall & Gal (2018): L = L_ce + 0.5*exp(-s1)*L1 + 0.5*s1 + 0.5*exp(-s2)*L2 + 0.5*s2
                s1 = self.log_var_coarse
                s2 = self.log_var_fine
                w1 = 0.5 * torch.exp(-s1)
                w2 = 0.5 * torch.exp(-s2)
                total_loss = total_loss + w1 * coarse_loss + 0.5 * s1 + w2 * fine_loss + 0.5 * s2
                dyn_cw = w1.detach()
                dyn_fw = w2.detach()
            else:
                total_loss = total_loss + lambda_coarse * coarse_loss + lambda_fine * fine_loss

            # KD
            if kd_teachers is not None and lambda_kd > 0:
                def kd_loss_fn(student_logits, teacher_logits, T=3.0):
                    if student_logits is None or teacher_logits is None: return None
                    if student_logits.size(-1) != teacher_logits.size(-1): return None
                    p_s = F.log_softmax(student_logits / T, dim=-1)
                    with torch.no_grad():
                        if torch.all(teacher_logits >= 0) and torch.all(teacher_logits <= 1):
                            p_t = teacher_logits / (teacher_logits.sum(dim=-1, keepdim=True) + 1e-8)
                        else:
                            p_t = F.softmax(teacher_logits / T, dim=-1)
                    return F.kl_div(p_s, p_t, reduction="batchmean") * (T*T)

                kd_vals = []
                if "tf" in kd_teachers:
                    v = kd_loss_fn(out["kd_tf_logits"], kd_teachers["tf"], T=getattr(self.cfg, "temp", 3.0))
                    if v is not None: kd_vals.append(v)
                if "in" in kd_teachers:
                    v = kd_loss_fn(out["kd_in_logits"], kd_teachers["in"], T=getattr(self.cfg, "temp", 3.0))
                    if v is not None: kd_vals.append(v)
                if "it" in kd_teachers:
                    v = kd_loss_fn(out["kd_it_logits"], kd_teachers["it"], T=getattr(self.cfg, "temp", 3.0))
                    if v is not None: kd_vals.append(v)
                if len(kd_vals) > 0:
                    kd_loss = sum(kd_vals)
                    total_loss = total_loss + lambda_kd * kd_loss

            # logic (off for 3-class)
            s = self.logic_score(out["kd_tf_logits"], out["kd_in_logits"], out["kd_it_logits"])
            if s is not None and lambda_logic > 0 and self.num_classes == 2:
                target = (labels == 0).float()
                logic_loss = F.mse_loss(s, target)
                total_loss = total_loss + lambda_logic * logic_loss

            # adversarial (weak / warm)
            if lambda_adv > 0:
                adv_it = F.cross_entropy(self.adv_it(self.grl(d_it)), labels)
                adv_tf = F.cross_entropy(self.adv_tf(self.grl(d_tf)), labels)
                adv_in = F.cross_entropy(self.adv_in(self.grl(d_in)), labels)
                adv_loss = (adv_it + adv_tf + adv_in) / 3.0
                total_loss = total_loss + lambda_adv * adv_loss

        out["loss"] = total_loss
        out["ce"] = ce_loss
        out["kd_loss"] = kd_loss
        out["coarse_loss"] = coarse_loss
        out["fine_loss"] = fine_loss
        out["logic_loss"] = logic_loss
        out["adv_loss"] = adv_loss
        if dyn_cw is not None: out["dyn_coarse_w"] = dyn_cw
        if dyn_fw is not None: out["dyn_fine_w"] = dyn_fw
        return out

# -----------------------------
# Full Model
# -----------------------------
@dataclass
class UpgConfig:
    img_out: int = 512
    txt_out: int = 768
    hidden: int = 512
    p_dim: int = 128
    num_classes: int = 3
    # SCABG
    K: int = 3
    num_queries: int = 8
    heads: int = 8
    contrastive_temp: float = 0.07
    lambda_contrast: float = 0.0
    lambda_band: float = 0.0
    gate_self_attn: bool = True
    # KD / logic / adv
    kd_dims: Tuple[int,int,int] = (2, 0, 2)
    entail_indices: Tuple[int,int,int] = (0, 0, 0)
    temp: float = 3.0
    lambda_kd: float = 0.03
    lambda_logic: float = 0.0
    lambda_adv: float = 0.001
    # Coarse/Fine (fallback weights; dynamic enabled by default)
    lambda_coarse: float = 0.3
    lambda_fine: float = 0.3
    dynamic_cf_weights: bool = True
    # Training aids
    ce_label_smoothing: float = 0.0
    class_weights: Optional[List[float]] = None
    roberta_name: str = "roberta-base"
    img_encoder: str = "google/vit-base-patch16-224"
    use_ctx: bool = True
    # Focal
    use_focal: bool = True
    focal_gamma: float = 2.0

class SCABG_LDT_Model(nn.Module):
    def __init__(self, cfg: UpgConfig):
        super().__init__()
        self.cfg = cfg
        self.img_enc = ImageEncoder(out_dim=cfg.hidden, encoder_type=cfg.img_encoder)
        self.txt_enc = TextEncoder(model_name=cfg.roberta_name, out_dim=cfg.txt_out)
        self.align_text = nn.Linear(cfg.txt_out, cfg.hidden)
        self.txt_map = MLP(cfg.hidden * 4, cfg.hidden, cfg.hidden)
        self.branch_it = SCABGBranch(cfg.hidden, cfg.K, cfg.num_queries, cfg.heads, cfg.p_dim, name="IT", gate_self_attn=cfg.gate_self_attn)
        self.branch_tf = SCABGBranch(cfg.hidden, cfg.K, cfg.num_queries, cfg.heads, cfg.p_dim, name="TF", gate_self_attn=cfg.gate_self_attn)
        self.branch_in = SCABGBranch(cfg.hidden, cfg.K, cfg.num_queries, cfg.heads, cfg.p_dim, name="IN", gate_self_attn=cfg.gate_self_attn)
        self.detector = DetectionHead(p_dim=cfg.p_dim, num_classes=cfg.num_classes,
                                      kd_dims=cfg.kd_dims, entail_indices=cfg.entail_indices,
                                      use_dynamic_cf=cfg.dynamic_cf_weights)
        self.detector.cfg = cfg

    def forward(self, image: torch.Tensor,
                news_ids: torch.Tensor, news_mask: torch.Tensor,
                ctx_ids: Optional[torch.Tensor] = None, ctx_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                teacher_tf: Optional[torch.Tensor] = None,
                teacher_in: Optional[torch.Tensor] = None,
                teacher_it: Optional[torch.Tensor] = None,
                coarse: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        img_out = self.img_enc(image)
        if img_out.dim() == 4:
            B, D, H, W = img_out.shape
            tok_img = img_out.flatten(2).transpose(1, 2)
        else:
            tok_img = img_out  # (B,N,hidden)

        news_seq, _ = self.txt_enc(news_ids, news_mask)
        news_seq_h = self.align_text(news_seq)
        cond = news_seq_h[:, 0]

        if ctx_ids is not None and ctx_mask is not None and self.cfg.use_ctx:
            ctx_seq, _ = self.txt_enc(ctx_ids, ctx_mask)
            ctx_seq_h = self.align_text(ctx_seq)
        else:
            ctx_seq_h = torch.zeros_like(news_seq_h)

        news_fft = dft_1d(news_seq_h)
        ctx_fft  = dft_1d(ctx_seq_h)
        _ = self.txt_map(torch.cat([news_fft.mean(dim=1), ctx_fft.mean(dim=1)], dim=-1))

        p_it, aux_it = self.branch_it(tok_img, ctx_seq_h, cond, return_band_info=True)
        p_tf, aux_tf = self.branch_tf(news_seq_h, ctx_seq_h, cond, return_band_info=True)
        p_in, aux_in = self.branch_in(tok_img, news_seq_h, cond, return_band_info=True)

        kd_teachers = {}
        if teacher_tf is not None: kd_teachers["tf"] = teacher_tf
        if teacher_in is not None: kd_teachers["in"] = teacher_in
        if teacher_it is not None: kd_teachers["it"] = teacher_it

        out = self.detector(
            p_it=p_it, p_tf=p_tf, p_in=p_in,
            labels=labels, coarse_labels=coarse,
            kd_teachers=kd_teachers if len(kd_teachers) > 0 else None
        )

        lambda_contrast = getattr(self.cfg, "lambda_contrast_curr", self.cfg.lambda_contrast)
        lambda_band     = getattr(self.cfg, "lambda_band_curr", self.cfg.lambda_band)
        contrastive_loss = None
        band_consistency = None

        if lambda_contrast > 0:
            loss1 = info_nce(p_tf, p_in, temperature=self.cfg.contrastive_temp)
            loss2 = info_nce(p_it, p_tf, temperature=self.cfg.contrastive_temp)
            contrastive_loss = 0.5 * (loss1 + loss2)
            if out["loss"] is not None:
                out["loss"] = out["loss"] + lambda_contrast * contrastive_loss

        if lambda_band > 0 and aux_it is not None and aux_tf is not None and aux_in is not None:
            g_it, g_tf, g_in = aux_it["band_gate"], aux_tf["band_gate"], aux_in["band_gate"]
            band_consistency = F.mse_loss(g_it, g_tf) + F.mse_loss(g_it, g_in) + F.mse_loss(g_tf, g_in)
            if out["loss"] is not None:
                out["loss"] = out["loss"] + lambda_band * band_consistency

        out["contrastive_loss"] = contrastive_loss
        out["band_consistency"] = band_consistency
        return out
