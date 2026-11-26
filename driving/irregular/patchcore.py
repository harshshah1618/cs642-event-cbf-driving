"""PatchCore-style novelty scores for camera frames (Roth et al., CVPR 2022).

A frozen pretrained encoder turns each frame into a grid of patch features. The
memory bank is a greedy coreset of patch features from clean training frames.
A patch's anomaly score is its distance to the nearest bank feature; the frame
score is the largest patch score. Nothing is trained.
"""

import math

import torch
import torch.nn.functional as F

MOSAIC_FRONT_ROWS = 252  # Harsh's capture: front camera 448x252 above rear 448x126
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def front_view(image):
    """Crop the forward camera out of the front/rear mosaic."""
    return image.crop((0, 0, image.width, MOSAIC_FRONT_ROWS))


def to_tensor(images, size):
    """PIL images -> normalised float tensor (B, 3, H, W) at `size` = (H, W)."""
    import numpy as np

    arrays = [np.asarray(im.convert("RGB").resize((size[1], size[0])), dtype=np.float32) / 255.0
              for im in images]
    x = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


class Encoder:
    """Frozen backbone returning patch features (B, H, W, D)."""

    def __init__(self, name="dinov2_small", device="cuda", dim=384):
        self.name, self.device, self.dim = name, device, dim
        if name == "dinov2_small":
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained("facebook/dinov2-small")
            self.size = (252, 448)  # 18 x 32 patches of 14 px
        elif name == "wrn50":
            from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

            self.model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            self.size = (256, 448)  # layer2 grid 32 x 56
        else:
            raise ValueError(f"Unknown encoder {name!r}")
        self.model.eval().to(device)

    @torch.inference_mode()
    def __call__(self, images):
        x = to_tensor(images, self.size).to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            if self.name == "dinov2_small":
                out = self.model(pixel_values=x).last_hidden_state[:, 1:, :]
                h, w = self.size[0] // 14, self.size[1] // 14
                feats = out.reshape(len(images), h, w, -1)
            else:
                m = self.model
                y = m.maxpool(m.relu(m.bn1(m.conv1(x))))
                l2 = m.layer2(m.layer1(y))
                l3 = m.layer3(l2)
                # PatchCore: 3x3 local neighbourhood averaging, layer3 upsampled to layer2's grid.
                l2 = F.avg_pool2d(l2, 3, 1, 1)
                l3 = F.interpolate(F.avg_pool2d(l3, 3, 1, 1), size=l2.shape[-2:],
                                   mode="bilinear", align_corners=False)
                f = torch.cat([l2, l3], 1).permute(0, 2, 3, 1)  # (B, H, W, 1536)
                b, h, w, c = f.shape
                f = F.adaptive_avg_pool1d(f.reshape(-1, 1, c).float(), self.dim)
                feats = f.reshape(b, h, w, self.dim)
        return feats.float()


def greedy_coreset(features, size, projection_dim=128, seed=0):
    """Indices of a greedy k-center coreset (approximate, on a random projection)."""
    n = features.shape[0]
    if size >= n:
        return torch.arange(n, device=features.device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    proj = torch.randn(features.shape[1], projection_dim, generator=gen) / math.sqrt(projection_dim)
    z = features.float() @ proj.to(features.device)
    first = int(torch.randint(n, (1,), generator=gen))
    chosen = [first]
    nearest = torch.linalg.norm(z - z[first], dim=1)
    for _ in range(size - 1):
        idx = int(torch.argmax(nearest))
        chosen.append(idx)
        nearest = torch.minimum(nearest, torch.linalg.norm(z - z[idx], dim=1))
    return torch.tensor(chosen, device=features.device)


def nearest_distance(queries, bank, chunk=4096):
    """Euclidean distance from each query row to its nearest bank row."""
    out = []
    bank = bank.float()
    bank_sq = (bank * bank).sum(1)
    for i in range(0, queries.shape[0], chunk):
        q = queries[i:i + chunk].float()
        d2 = (q * q).sum(1, keepdim=True) - 2 * q @ bank.T + bank_sq
        out.append(d2.clamp_min(0).min(1).values.sqrt())
    return torch.cat(out)


class PatchCore:
    def __init__(self, bank, encoder_name, grid):
        self.bank, self.encoder_name, self.grid = bank, encoder_name, tuple(grid)

    @classmethod
    def fit(cls, encoder, batches, patches_per_image=64, coreset_size=20000, seed=0):
        """`batches` yields lists of PIL front-camera images from clean frames."""
        gen = torch.Generator(device="cpu").manual_seed(seed)
        pool = []
        grid = None
        for images in batches:
            f = encoder(images)
            grid = f.shape[1:3]
            flat = f.reshape(f.shape[0], -1, f.shape[-1])
            for row in flat:
                pick = torch.randperm(row.shape[0], generator=gen)[:patches_per_image]
                pool.append(row[pick.to(row.device)].half())
        features = torch.cat(pool)
        print(f"[coreset] selecting {coreset_size} of {features.shape[0]} patches", flush=True)
        keep = greedy_coreset(features, coreset_size, seed=seed)
        return cls(features[keep].float(), encoder.name, grid), features.shape[0]

    def score(self, features):
        """features (B, H, W, D) -> frame scores (B,), patch maps (B, H, W)."""
        b, h, w, d = features.shape
        dist = nearest_distance(features.reshape(-1, d), self.bank.to(features.device))
        maps = dist.reshape(b, h, w)
        return maps.flatten(1).max(1).values, maps

    def save(self, path):
        torch.save(dict(bank=self.bank.cpu(), encoder=self.encoder_name, grid=self.grid), path)

    @classmethod
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location=device)
        return cls(state["bank"].to(device), state["encoder"], state["grid"])


def local_contrast_metrics(patch_map):
    """Separate localized physical hazards from uniform global weather shifts (rain/fog).

    Returns:
        dict with keys: max_score, median_score, contrast_ratio, cx_norm, cy_norm.
    """
    import numpy as np

    arr = np.asarray(patch_map, dtype=np.float64)
    if arr.size == 0:
        return dict(max_score=0.0, median_score=0.0, contrast_ratio=1.0, cx_norm=0.5, cy_norm=0.5)
    max_s = float(np.max(arr))
    med_s = float(np.median(arr))
    contrast = max_s / max(med_s, 1e-6)
    r, c = np.unravel_index(int(np.argmax(arr)), arr.shape)
    h, w = arr.shape
    return dict(
        max_score=max_s,
        median_score=med_s,
        contrast_ratio=float(contrast),
        cx_norm=float((c + 0.5) / max(w, 1)),
        cy_norm=float((r + 0.5) / max(h, 1)),
    )
