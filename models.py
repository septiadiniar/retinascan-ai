"""
models.py — Definisi model, inference, Attention Rollout, dan Grad-CAM
Untuk Streamlit app: Klasifikasi Penyakit Retina OCT
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import timm
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# Konfigurasi global
# ──────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'MH', 'NORMAL']
NUM_CLASSES = len(CLASS_NAMES)

CLASS_INFO = {
    'AMD'    : ('Age-related Macular Degeneration', '🟠', 'Degenerasi makula terkait usia, ditandai drusen dan atrofi'),
    'CNV'    : ('Choroidal Neovascularization',     '🔴', 'Pertumbuhan pembuluh darah abnormal di bawah retina'),
    'CSR'    : ('Central Serous Retinopathy',       '🟡', 'Penumpukan cairan di bawah retina sentral'),
    'DME'    : ('Diabetic Macular Edema',           '🔴', 'Pembengkakan makula akibat komplikasi diabetes'),
    'DR'     : ('Diabetic Retinopathy',             '🔴', 'Kerusakan retina akibat diabetes mellitus'),
    'DRUSEN' : ('Drusen',                           '🟡', 'Endapan kuning di bawah retina, risiko AMD'),
    'MH'     : ('Macular Hole',                     '🟠', 'Lubang kecil di makula, memengaruhi penglihatan sentral'),
    'NORMAL' : ('Normal Retina',                    '🟢', 'Retina sehat tanpa tanda-tanda penyakit'),
}

SEVERITY = {
    'NORMAL' : ('Sehat',   '#22c55e'),
    'DRUSEN' : ('Ringan',  '#eab308'),
    'AMD'    : ('Sedang',  '#f97316'),
    'CSR'    : ('Sedang',  '#f97316'),
    'MH'     : ('Sedang',  '#f97316'),
    'CNV'    : ('Berat',   '#ef4444'),
    'DME'    : ('Berat',   '#ef4444'),
    'DR'     : ('Berat',   '#ef4444'),
}

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ──────────────────────────────────────────────
# Arsitektur Model
# ──────────────────────────────────────────────
class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_base_patch14_dinov2',
            pretrained=False,
            num_classes=0,
            img_size=IMG_SIZE
        )
        embed_dim = self.backbone.embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def build_resnet50(num_classes=NUM_CLASSES):
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
    return m


def build_efficientnet_b3(num_classes=NUM_CLASSES):
    m = models.efficientnet_b3(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.classifier[1].in_features, num_classes)
    )
    return m


def build_vit_supervised(num_classes=NUM_CLASSES):
    return timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)


MODEL_BUILDERS = {
    'DINOv2 ViT-B/14 (Utama)' : DINOv2Classifier,
    'ResNet-50'                : build_resnet50,
    'EfficientNet-B3'          : build_efficientnet_b3,
    'ViT-B/16 Supervised'      : build_vit_supervised,
}

MODEL_FILES = {
    'DINOv2 ViT-B/14 (Utama)' : 'dinov2_best.pth',
    'ResNet-50'                : 'ResNet-50.pth',
    'EfficientNet-B3'          : 'EfficientNet-B3.pth',
    'ViT-B/16 Supervised'      : 'ViT-B-16_Supervised.pth',
}

MODEL_DESC = {
    'DINOv2 ViT-B/14 (Utama)' : '86M params | Self-supervised pretraining | Attention Rollout tersedia',
    'ResNet-50'                : '25M params | CNN klasik | Grad-CAM tersedia',
    'EfficientNet-B3'          : '12M params | Compound scaling | Grad-CAM tersedia',
    'ViT-B/16 Supervised'      : '86M params | Supervised ImageNet | Attention Rollout tersedia',
}


# ──────────────────────────────────────────────
# Load Model
# ──────────────────────────────────────────────
def load_model(model_name: str, checkpoint_path: str | None = None):
    """
    Load model dari checkpoint (.pth).
    Jika tidak ada checkpoint → demo mode dengan random weights (CEPAT, tanpa download).
    """
    builder = MODEL_BUILDERS[model_name]
    model   = builder()
    model   = model.to(DEVICE)

    if checkpoint_path:
        try:
            state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state, strict=False)
            model.eval()
            return model, True
        except Exception as e:
            pass

    # Demo mode: pakai random weights — TIDAK download apapun, langsung siap
    model.eval()
    return model, False


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────
@torch.no_grad()
def predict(model, pil_image: Image.Image):
    """
    Returns:
        pred_class  : str nama kelas
        confidence  : float 0-1
        all_probs   : dict {class_name: prob}
        top3        : list of (class_name, prob) sorted desc
    """
    tensor  = val_transform(pil_image).unsqueeze(0).to(DEVICE)
    logits  = model(tensor)
    probs   = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx    = probs.argmax()
    pred_class  = CLASS_NAMES[pred_idx]
    confidence  = float(probs[pred_idx])
    all_probs   = {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}
    top3        = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]

    return pred_class, confidence, all_probs, top3


# ──────────────────────────────────────────────
# Attention Rollout (untuk ViT-based models)
# ──────────────────────────────────────────────
class AttentionRollout:
    def __init__(self, model, discard_ratio=0.9):
        self.model         = model
        self.discard_ratio = discard_ratio
        self.attentions    = []
        self.hooks         = []

        # Cari blok attention (timm ViT)
        backbone = getattr(model, 'backbone', model)
        if hasattr(backbone, 'blocks'):
            for block in backbone.blocks:
                h = block.attn.register_forward_hook(self._hook)
                self.hooks.append(h)

    def _hook(self, module, input, output):
        x = input[0]
        B, N, C = x.shape
        nh = module.num_heads
        hd = C // nh
        with torch.no_grad():
            qkv = module.qkv(x).reshape(B, N, 3, nh, hd).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1) * (hd ** -0.5)).softmax(dim=-1)
        self.attentions.append(attn.detach().cpu())

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def __call__(self, pil_image: Image.Image):
        self.attentions = []
        tensor = val_transform(pil_image).unsqueeze(0).to(DEVICE)

        self.model.eval()
        with torch.no_grad():
            self.model(tensor)

        if not self.attentions:
            return None

        result = torch.eye(self.attentions[0].shape[-1])
        for attn in self.attentions:
            a    = attn[0].mean(0)          # avg over heads: (N, N)
            flat = a.flatten()
            flat[flat.argsort()[:int(len(flat) * self.discard_ratio)]] = 0
            a    = a + torch.eye(a.shape[0])
            a    = a / a.sum(-1, keepdim=True)
            result = a @ result

        mask = result[0, 1:]               # skip CLS token
        g    = int(mask.shape[0] ** 0.5)
        mask = mask.reshape(g, g).numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask


def rollout_to_heatmap(mask: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Convert attention mask → RGB heatmap (H, W, 3) float32 [0,1]."""
    import matplotlib.cm as cm
    upsampled = np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    ) / 255.0
    heatmap = cm.jet(upsampled)[:, :, :3]
    return heatmap.astype(np.float32)


# ──────────────────────────────────────────────
# Grad-CAM (untuk CNN dan ViT)
# ──────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, model_name: str):
        self.model      = model
        self.model_name = model_name
        self.gradients  = None
        self.activations = None
        self._register(model_name)

    def _register(self, model_name):
        target = self._get_target_layer(model_name)
        if target is None:
            return
        target.register_forward_hook(self._save_act)
        target.register_backward_hook(self._save_grad)

    def _get_target_layer(self, model_name):
        model = self.model
        try:
            if 'ResNet' in model_name:
                return model.layer4[-1]
            elif 'EfficientNet' in model_name:
                return model.features[-1]
            elif 'DINOv2' in model_name:
                return list(model.backbone.blocks)[-1].norm1
            elif 'ViT' in model_name:
                return list(model.blocks)[-1].norm1
        except:
            return None

    def _save_act(self, m, i, o):
        self.activations = o.detach()

    def _save_grad(self, m, gi, go):
        self.gradients = go[0].detach()

    def __call__(self, pil_image: Image.Image, class_idx: int | None = None):
        self.model.eval()
        tensor = val_transform(pil_image).unsqueeze(0).to(DEVICE)
        tensor.requires_grad_(True)

        output = self.model(tensor)
        if class_idx is None:
            class_idx = output.argmax(1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            return None

        grad = self.gradients[0]      # (N, C) or (C, H, W)
        act  = self.activations[0]

        # ViT case: tokens
        if grad.dim() == 2:
            weights = grad.mean(-1)
            cam     = (weights.unsqueeze(-1) * act).sum(-1)
            cam     = cam[1:] if cam.shape[0] > 1 else cam   # skip CLS
            g       = int(cam.shape[0] ** 0.5)
            cam     = cam.reshape(g, g)
        else:
            # CNN case: (C, H, W)
            weights = grad.mean(dim=(-2, -1))
            cam     = (weights[:, None, None] * act).sum(0)

        cam = F.relu(cam).cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def cam_to_heatmap(cam: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Convert CAM → RGB heatmap [0,1]."""
    import matplotlib.cm as cm
    upsampled = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    ) / 255.0
    return cm.jet(upsampled)[:, :, :3].astype(np.float32)


def overlay_heatmap(orig_np: np.ndarray, heatmap: np.ndarray, alpha=0.5) -> np.ndarray:
    """Overlay heatmap di atas gambar asli."""
    if orig_np.max() > 1:
        orig_np = orig_np / 255.0
    return np.clip(alpha * orig_np + (1 - alpha) * heatmap, 0, 1)
