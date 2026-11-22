"""Camera-side weather overlays (MetaDrive has no weather).

Rain only changes what the camera sees; wet-road physics is a separate knob
(reduced brake force, see scene.Placement.brake_scale). Overlays are seeded per
frame so runs are reproducible.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def rain(image, seed=0, streaks=450, darken=0.72):
    """Darker, flatter colours, rain streaks and a slight blur."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    grey = arr.mean(axis=2, keepdims=True)
    arr = (0.7 * arr + 0.3 * grey) * darken + 18  # desaturate, darken, lift blacks (haze)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    layer = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    w, h = out.size
    for _ in range(streaks):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        length = rng.uniform(8, 22)
        dx = length * 0.25
        alpha = int(rng.uniform(60, 130))
        draw.line([(x, y), (x + dx, y + length)], fill=(210, 215, 225, alpha), width=1)
    out = Image.alpha_composite(out.convert("RGBA"), layer).convert("RGB")
    return out.filter(ImageFilter.GaussianBlur(0.7))


def fog(image, density=0.55):
    """Distance-like fog: blend toward grey, stronger near the horizon (upper half)."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h = arr.shape[0]
    rows = np.linspace(1.0, 0.25, h)[:, None, None]  # top rows (far) fade most
    weight = np.clip(density * rows * 1.3, 0, 0.9)
    arr = arr * (1 - weight) + 200 * weight
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


OVERLAYS = {"rain": rain, "fog": lambda image, seed=0: fog(image)}
