"""Custom road props for irregular scenarios: signs on a pole with swappable faces.

MetaDrive has no sign objects, so a sign is a static physics box (it registers
collisions and appears in the 50 m object query) plus an unlit textured card
drawn with PIL. The face can be swapped mid-episode (a flagger's STOP -> SLOW).
"""

import io
from functools import lru_cache

# Face name -> (width m, height m). Sizes follow common US work-zone signs.
FACES = {
    "stop": (0.6, 0.6),          # flagger paddle (18-24 in), red octagon
    "slow": (0.6, 0.6),          # paddle back side, orange diamond
    "lane_closed": (1.2, 1.2),   # RIGHT LANE CLOSED AHEAD, orange diamond
    "billboard": (1.6, 0.9),     # harmless roadside board (control)
}


@lru_cache(maxsize=None)
def face_png(face):
    """Render a sign face to PNG bytes (RGBA, transparent outside the shape)."""
    from PIL import Image, ImageDraw, ImageFont

    w_m, h_m = FACES[face]
    w, h = int(256 * w_m / max(w_m, h_m)), int(256 * h_m / max(w_m, h_m))
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def font(size):
        try:
            return ImageFont.load_default(size=size)
        except TypeError:  # Pillow < 10.1
            return ImageFont.load_default()

    def centred(lines, size, fill):
        f = font(size)
        boxes = [d.textbbox((0, 0), t, font=f) for t in lines]
        total = sum(b[3] - b[1] for b in boxes) + 6 * (len(lines) - 1)
        y = (h - total) / 2
        for t, b in zip(lines, boxes, strict=True):
            d.text(((w - (b[2] - b[0])) / 2 - b[0], y - b[1]), t, font=f, fill=fill)
            y += b[3] - b[1] + 6

    if face == "stop":
        c, r = w / 2, w / 2 - 2
        import math
        pts = [(c + r * math.cos(math.radians(22.5 + 45 * i)),
                c + r * math.sin(math.radians(22.5 + 45 * i))) for i in range(8)]
        d.polygon(pts, fill=(200, 16, 30, 255), outline=(255, 255, 255, 255), width=8)
        centred(["STOP"], 78, (255, 255, 255, 255))
    elif face in ("slow", "lane_closed"):
        d.polygon([(w / 2, 2), (w - 2, h / 2), (w / 2, h - 2), (2, h / 2)],
                  fill=(255, 128, 0, 255), outline=(20, 20, 20, 255), width=6)
        if face == "slow":
            centred(["SLOW"], 64, (15, 15, 15, 255))
        else:
            centred(["RIGHT", "LANE", "CLOSED", "AHEAD"], 34, (15, 15, 15, 255))
    elif face == "billboard":
        d.rectangle([0, 0, w - 1, h - 1], fill=(240, 240, 235, 255), outline=(60, 60, 60, 255), width=6)
        centred(["FRESH", "PEACHES 2 MI"], 30, (30, 70, 140, 255))
    else:
        raise ValueError(f"Unknown face {face!r}")
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()


def make_sign_class():
    from metadrive.component.static_object.traffic_object import TrafficObject
    from metadrive.constants import MetaDriveType
    from metadrive.engine.physics_node import BaseRigidBodyNode
    from panda3d.bullet import BulletBoxShape
    from panda3d.core import CardMaker, PNMImage, StringStream, Texture, TransparencyAttrib

    textures = {}

    def texture(face):
        if face not in textures:
            image = PNMImage()
            image.read(StringStream(face_png(face)), f"{face}.png")
            tex = Texture(face)
            tex.load(image)
            textures[face] = tex
        return textures[face]

    class RoadSign(TrafficObject):
        CLASS_NAME = MetaDriveType.TRAFFIC_OBJECT
        HEIGHT = 2.2   # top of the face above the ground (m)
        MASS = 10

        def __init__(self, position, heading_theta, lane=None, face="stop", static=True,
                     random_seed=None, name=None):
            super().__init__(position, heading_theta, lane, random_seed, name)
            w, h = FACES[face]
            self.face_name = face
            n = BaseRigidBodyNode(self.name, self.CLASS_NAME)
            self.add_body(n)
            self.body.addShape(BulletBoxShape((0.15, max(w, 0.2) / 2, self.HEIGHT / 2)))
            self.set_static(static)
            self._face_np = None
            if self.render:
                z_top = self.HEIGHT / 2
                pole = CardMaker("pole")
                pole.setFrame(-0.04, 0.04, -self.HEIGHT / 2, z_top - h)
                pole.setColor(0.35, 0.35, 0.35, 1)
                for angle in (0, 90):  # two crossed cards read as a round pole
                    p = self.origin.attachNewNode(pole.generate())
                    p.setH(angle)
                    p.setTwoSided(True)
                    p.setLightOff()
                    # MetaDrive's scene shader ignores vertex colours and card textures
                    # (renders them black or fully transparent); use fixed-function.
                    p.setShaderOff(1)
                card = CardMaker("face")
                card.setFrame(-w / 2, w / 2, z_top - h, z_top)
                self._face_np = self.origin.attachNewNode(card.generate())
                # MetaDrive's local x axis points along the road; a card lies in the x-z
                # plane, so turn it -90 degrees to face approaching traffic.
                self._face_np.setH(-90)
                self._face_np.setTwoSided(True)
                self._face_np.setLightOff()
                self._face_np.setTransparency(TransparencyAttrib.MAlpha)
                self._face_np.setShaderOff(1)
                self.set_face(face)

        def reset(self, *args, **kwargs):
            raise RuntimeError("RoadSign is spawned with force_spawn=True; never recycled")

        def set_face(self, face):
            self.face_name = face
            if self._face_np is not None:
                self._face_np.setTexture(texture(face), 1)

        @property
        def top_down_length(self):
            return 0.3

        @property
        def top_down_width(self):
            return FACES[self.face_name][0]

        @property
        def LENGTH(self):
            return 0.3

        @property
        def WIDTH(self):
            return FACES[self.face_name][0]

    return RoadSign
