import os
import time
import math
import random
import threading
import urllib.request
import cv2
import numpy as np
import sounddevice as sd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully.")


# -----------------------------------------------------------------------------
# ZERO-LATENCY PROCEDURAL SOUND ENGINE (DUAL DOMAIN SUITE)
# -----------------------------------------------------------------------------
class DualDomainSoundEngine:
    """Synthesizes high-fidelity audio for both Gojo and Sukuna's Domains."""
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        self.sounds = {
            "gojo_expand": self._synth_gojo_expand(),
            "sukuna_expand": self._synth_sukuna_expand(),
            "shatter": self._synth_domain_shatter(),
            "slash": self._synth_slash_sfx(),
        }

    def play(self, sound_name):
        """Play sound asynchronously without delaying video frames."""
        if sound_name in self.sounds:
            audio = self.sounds[sound_name]
            threading.Thread(target=self._play_worker, args=(audio,), daemon=True).start()

    def _play_worker(self, data):
        try:
            sd.play(data, self.sr)
            sd.wait()
        except Exception:
            pass

    def _synth_gojo_expand(self):
        """Gojo Infinite Void: Deep resonant bass + celestial bells + rising vacuum sweep."""
        duration = 3.2
        t = np.linspace(0, duration, int(self.sr * duration), False)
        f_drone = 90.0 - 42.0 * (t / duration)
        phase_drone = 2.0 * np.pi * np.cumsum(f_drone) / self.sr
        drone = np.sin(phase_drone) * np.clip(t / 0.6, 0, 1) * np.exp(-t * 0.25)

        f_bells = [528.0, 1056.0, 1584.0, 2112.0, 3168.0]
        bell = np.zeros_like(t)
        for i, fb in enumerate(f_bells):
            bell += (1.0 / (i + 1.2)) * np.sin(2.0 * np.pi * fb * t) * np.exp(-t * (1.2 + i * 0.4))

        sweep_f = 100.0 * np.exp(t * 1.3)
        sweep = np.sin(2.0 * np.pi * np.cumsum(sweep_f) / self.sr) * np.clip(t / 1.2, 0, 1) * np.exp(-t * 0.7)
        sub = np.sin(2.0 * np.pi * 42.0 * t) * np.exp(-t * 0.4)

        mix = drone * 0.60 + bell * 0.45 + sweep * 0.35 + sub * 0.55
        return (np.tanh(mix * 1.8) * 0.92).astype(np.float32)

    def _synth_sukuna_expand(self):
        """Sukuna Malevolent Shrine: Demonic sub-bass roar + sinister temple bell & dark energy surge."""
        duration = 3.2
        t = np.linspace(0, duration, int(self.sr * duration), False)

        # Demonic sub-bass growl & cursed energy resonance
        f0 = 65.0 - 24.0 * (t / duration)
        phase = 2.0 * np.pi * np.cumsum(f0) / self.sr
        buzz = np.sin(phase) + 0.8 * np.sin(2 * phase) + 0.5 * np.sin(3 * phase) + 0.3 * np.sin(4 * phase)
        sub = buzz * np.exp(-t * 0.35)

        # Sinister deep temple bell & gong resonance
        gong = (np.sin(2 * np.pi * 108.0 * t) * np.exp(-t * 0.75) +
                0.6 * np.sin(2 * np.pi * 216.0 * t) * np.exp(-t * 1.1) +
                0.35 * np.sin(2 * np.pi * 324.0 * t) * np.exp(-t * 1.5))

        # Rising dark vacuum sweep
        sweep_f = 80.0 * np.exp(t * 1.2)
        sweep = np.sin(2.0 * np.pi * np.cumsum(sweep_f) / self.sr) * np.clip(t / 1.0, 0, 1) * np.exp(-t * 0.6)

        mix = sub * 0.65 + gong * 0.60 + sweep * 0.35
        return (np.tanh(mix * 1.8) * 0.95).astype(np.float32)

    def _synth_domain_shatter(self):
        """Domain barrier shattering into obsidian glass shards."""
        duration = 1.4
        t = np.linspace(0, duration, int(self.sr * duration), False)
        noise = np.random.uniform(-1.0, 1.0, len(t))
        shatter = noise * np.exp(-t * 7.5) * 0.75
        ping = (np.sin(2.0 * np.pi * 3400.0 * t) * np.exp(-t * 5.5) * 0.45 +
                np.sin(2.0 * np.pi * 5100.0 * t) * np.exp(-t * 8.5) * 0.35)
        sub = np.sin(2.0 * np.pi * (85.0 * np.exp(-t * 3.5)) * t) * np.exp(-t * 2.2) * 0.75
        mix = shatter * 0.60 + ping * 0.50 + sub * 0.75
        return (np.tanh(mix * 1.7) * 0.88).astype(np.float32)

    def _synth_slash_sfx(self):
        """High-speed blade cut (Cleave / Dismantle)."""
        duration = 0.22
        t = np.linspace(0, duration, int(self.sr * duration), False)
        f = 5400.0 * np.exp(-t * 24.0) + 140.0
        s = np.sin(2.0 * np.pi * np.cumsum(f) / self.sr) * np.exp(-t * 14.0)
        noise = np.random.uniform(-1, 1, len(t)) * np.exp(-t * 16.0) * 0.35
        return (np.tanh((s * 0.8 + noise) * 1.5) * 0.75).astype(np.float32)


# -----------------------------------------------------------------------------
# ONE-EURO ADAPTIVE SMOOTHING FILTER
# -----------------------------------------------------------------------------
class OneEuroFilter:
    def __init__(self, min_cutoff=1.1, beta=0.04, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t=None):
        if t is None:
            t = time.time()
        if self.x_prev is None:
            self.x_prev = float(x)
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        dt = max(1e-4, t - self.t_prev)
        self.t_prev = t

        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat


class OneEuroPointFilter:
    def __init__(self, min_cutoff=1.1, beta=0.04):
        self.fx = OneEuroFilter(min_cutoff, beta)
        self.fy = OneEuroFilter(min_cutoff, beta)

    def update(self, pt):
        if pt is None:
            return None
        t = time.time()
        return int(self.fx.filter(pt[0], t)), int(self.fy.filter(pt[1], t))

    def reset(self):
        self.fx.x_prev = None
        self.fy.x_prev = None

class HandLandmarkSmoother:
    def __init__(self, min_cutoff=1.0, beta=0.015):
        self.fx = [OneEuroFilter(min_cutoff, beta) for _ in range(21)]
        self.fy = [OneEuroFilter(min_cutoff, beta) for _ in range(21)]

    def update(self, landmarks):
        if landmarks is None:
            return None
        smoothed = []
        t = time.time()
        for i, lm in enumerate(landmarks):
            sx = self.fx[i].filter(lm.x, t)
            sy = self.fy[i].filter(lm.y, t)
            class SmoothedLM:
                def __init__(self, x, y, z):
                    self.x = x
                    self.y = y
                    self.z = z
            smoothed.append(SmoothedLM(sx, sy, getattr(lm, 'z', 0.0)))
        return smoothed

    def reset(self):
        for f in self.fx:
            f.x_prev = None
        for f in self.fy:
            f.x_prev = None


# -----------------------------------------------------------------------------
# GRAPHICS & SKELETON
# -----------------------------------------------------------------------------
_glow_cache = {}

def get_glow_sprite(radius, color_bgr):
    key = (int(radius), color_bgr)
    if key in _glow_cache:
        return _glow_cache[key]

    r = int(max(4, radius))
    size = r * 2 + 1
    sprite = np.zeros((size, size, 3), dtype=np.uint8)

    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - r) ** 2 + (y - r) ** 2)
    mask = np.clip(1.0 - (dist / r), 0, 1) ** 1.8

    for c in range(3):
        sprite[:, :, c] = np.clip(color_bgr[c] * mask, 0, 255).astype(np.uint8)

    if len(_glow_cache) > 250:
        _glow_cache.clear()

    _glow_cache[key] = sprite
    return sprite


def blend_glow(img, center, radius, color_bgr):
    if radius <= 2:
        return
    sprite = get_glow_sprite(radius, color_bgr)
    h, w = img.shape[:2]
    sh, sw = sprite.shape[:2]
    r = sh // 2
    cx, cy = int(center[0]), int(center[1])

    x1, y1 = cx - r, cy - r
    x2, y2 = x1 + sw, y1 + sh

    sx1 = max(0, -x1)
    sy1 = max(0, -y1)
    sx2 = sw - max(0, x2 - w)
    sy2 = sh - max(0, y2 - h)

    dx1 = max(0, x1)
    dy1 = max(0, y1)
    dx2 = min(w, x2)
    dy2 = min(h, y2)

    if dx2 <= dx1 or dy2 <= dy1 or sx2 <= sx1 or sy2 <= sy1:
        return

    img[dy1:dy2, dx1:dx2] = cv2.add(img[dy1:dy2, dx1:dx2], sprite[sy1:sy2, sx1:sx2])


def draw_lightning(img, pt1, pt2, color, thickness=2, segments=6, jitter=12):
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dist = math.hypot(dx, dy)
    if dist < 6:
        return

    nx = -dy / dist
    ny = dx / dist

    points = [pt1]
    for i in range(1, segments):
        t = i / segments
        px = int(pt1[0] + dx * t + nx * random.uniform(-jitter, jitter))
        py = int(pt1[1] + dy * t + ny * random.uniform(-jitter, jitter))
        points.append((px, py))
    points.append(pt2)

    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], color, thickness + 1, cv2.LINE_AA)
        cv2.line(img, points[i], points[i + 1], (255, 255, 255), max(1, thickness // 2), cv2.LINE_AA)


FINGER_CONNECTIONS = {
    "palm": [(0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (0, 17)],
    "thumb": [(1, 2), (2, 3), (3, 4)],
    "index": [(5, 6), (6, 7), (7, 8)],
    "middle": [(9, 10), (10, 11), (11, 12)],
    "ring": [(13, 14), (14, 15), (15, 16)],
    "pinky": [(17, 18), (18, 19), (19, 20)],
}

FINGER_COLORS = {
    "palm": (210, 120, 255),
    "thumb": (240, 50, 200),
    "index": (255, 235, 0),
    "middle": (80, 245, 120),
    "ring": (30, 215, 255),
    "pinky": (30, 130, 255),
}

FINGERTIP_INDICES = [4, 8, 12, 16, 20]
FINGERTIP_COLORS = [
    (240, 50, 200),
    (255, 235, 0),
    (80, 245, 120),
    (30, 215, 255),
    (30, 130, 255),
]


def draw_hand_skeleton(img, landmarks, width, height, domain_type=None):
    """Draw stylish neon skeleton, with celestial or demonic glow depending on domain."""
    pts = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

    # Bone colors
    for finger, connections in FINGER_CONNECTIONS.items():
        if domain_type == "GOJO":
            color = (255, 255, 255)
        elif domain_type == "SUKUNA":
            color = (30, 40, 255)  # Crimson blood
        else:
            color = FINGER_COLORS[finger]

        for start_idx, end_idx in connections:
            p1, p2 = pts[start_idx], pts[end_idx]
            cv2.line(img, p1, p2, color, 3 if not domain_type else 4, cv2.LINE_AA)

    # Joints
    for i, pt in enumerate(pts):
        if i not in FINGERTIP_INDICES and i != 0:
            cv2.circle(img, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, pt, 6, (60, 60, 60), 1, cv2.LINE_AA)

    wrist_col = (255, 200, 255) if domain_type == "GOJO" else ((40, 60, 255) if domain_type == "SUKUNA" else (210, 120, 255))
    cv2.circle(img, pts[0], 7, wrist_col, -1, cv2.LINE_AA)
    cv2.circle(img, pts[0], 10, (255, 255, 255), 2, cv2.LINE_AA)

    # Fingertip glow
    for idx, color in zip(FINGERTIP_INDICES, FINGERTIP_COLORS):
        pt = pts[idx]
        if domain_type == "GOJO":
            glow_c = (255, 240, 100)
        elif domain_type == "SUKUNA":
            glow_c = (30, 120, 255)  # Crimson fire
        else:
            glow_c = color
        cv2.circle(img, pt, 7, glow_c, -1, cv2.LINE_AA)
        cv2.circle(img, pt, 11, (255, 255, 255), 2, cv2.LINE_AA)


# -----------------------------------------------------------------------------
# GOJO'S INFINITE VOID: PURE SPACE & STARS SYSTEM
# -----------------------------------------------------------------------------
class BackgroundStar:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.x = random.randint(0, w - 1)
        self.y = random.randint(0, h - 1)
        self.size = random.choice([1, 1, 1, 1, 2, 2, 3])
        self.base_brightness = random.uniform(0.35, 1.0)
        self.twinkle_speed = random.uniform(1.8, 6.5)
        self.phase = random.uniform(0, 2 * math.pi)
        self.color = random.choice([
            (255, 250, 240),
            (255, 225, 190),
            (210, 190, 255),
            (190, 235, 255),
            (255, 255, 220),
        ])

    def draw(self, img, t):
        twinkle = self.base_brightness * (0.55 + 0.45 * math.sin(t * self.twinkle_speed + self.phase))
        b, g, r = self.color
        c = (int(b * twinkle), int(g * twinkle), int(r * twinkle))
        if self.size == 1:
            img[self.y, self.x] = c
        else:
            cv2.circle(img, (self.x, self.y), self.size, c, -1, cv2.LINE_AA)


class WarpStar:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.reset()

    def reset(self):
        self.x = random.uniform(-self.w * 0.7, self.w * 0.7)
        self.y = random.uniform(-self.h * 0.7, self.h * 0.7)
        self.z = random.uniform(0.8, 10.0)
        self.prev_z = self.z

    def update_and_draw(self, img, cx, cy, speed=0.18):
        self.prev_z = self.z
        self.z -= speed
        if self.z <= 0.12:
            self.reset()
            return

        px = int(cx + (self.x / self.z))
        py = int(cy + (self.y / self.z))
        old_px = int(cx + (self.x / self.prev_z))
        old_py = int(cy + (self.y / self.prev_z))

        if 0 <= px < self.w and 0 <= py < self.h:
            alpha = max(0.15, min(1.0, 1.0 - (self.z / 10.0)))
            b = int(255 * alpha)
            g = int(235 * alpha)
            r = int(210 * alpha)

            sz = max(1, int(3.2 / self.z))
            if sz == 1:
                cv2.line(img, (old_px, old_py), (px, py), (b, g, r), 1, cv2.LINE_AA)
            else:
                cv2.line(img, (old_px, old_py), (px, py), (b, g, r), sz, cv2.LINE_AA)
                cv2.circle(img, (px, py), sz, (255, 255, 255), -1, cv2.LINE_AA)


class ShootingStar:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.active = False
        self.reset()

    def reset(self):
        self.active = False
        self.timer = random.uniform(1.8, 5.0)
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.length = 0
        self.life = 0
        self.max_life = 1

    def spawn(self):
        self.active = True
        self.x = random.uniform(80, self.w - 80)
        self.y = random.uniform(20, self.h * 0.45)
        angle = random.uniform(math.pi * 0.18, math.pi * 0.36)
        speed = random.uniform(28.0, 42.0)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.length = random.uniform(70, 130)
        self.life = random.randint(14, 22)
        self.max_life = self.life
        self.color = random.choice([(255, 255, 255), (255, 235, 190), (210, 240, 255)])

    def update_and_draw(self, img, dt):
        if not self.active:
            self.timer -= dt
            if self.timer <= 0:
                self.spawn()
            return

        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        alpha = max(0.0, self.life / self.max_life)

        if self.life <= 0 or self.x >= self.w or self.y >= self.h:
            self.reset()
            return

        tail_x = int(self.x - (self.vx / 32.0) * self.length)
        tail_y = int(self.y - (self.vy / 32.0) * self.length)
        head_x = int(self.x)
        head_y = int(self.y)

        b, g, r = self.color
        col = (int(b * alpha), int(g * alpha), int(r * alpha))
        cv2.line(img, (tail_x, tail_y), (head_x, head_y), col, 2, cv2.LINE_AA)
        cv2.circle(img, (head_x, head_y), 3, (255, 255, 255), -1, cv2.LINE_AA)


# -----------------------------------------------------------------------------
# SUKUNA'S MALEVOLENT SHRINE: CLEAVE & DISMANTLE SYSTEM
# -----------------------------------------------------------------------------
class DismantleSlash:
    """High-speed Cleave & Dismantle razor slash cutting reality."""
    def __init__(self, w, h):
        cx = random.randint(40, w - 40)
        cy = random.randint(40, h - 40)
        length = random.uniform(160, 480)
        angle = random.uniform(0, math.pi)
        dx = math.cos(angle) * (length / 2)
        dy = math.sin(angle) * (length / 2)
        self.p1 = (int(cx - dx), int(cy - dy))
        self.p2 = (int(cx + dx), int(cy + dy))
        self.life = random.randint(5, 9)
        self.max_life = self.life
        self.thickness = random.randint(2, 5)

    def update_and_draw(self, img):
        self.life -= 1
        if self.life <= 0:
            return False
        alpha = self.life / self.max_life

        # Crimson blood outer edge
        c_outer = (int(30 * alpha), int(40 * alpha), int(255 * alpha))
        cv2.line(img, self.p1, self.p2, c_outer, self.thickness + 4, cv2.LINE_AA)

        # Pure white razor blade core
        c_inner = (int(255 * alpha), int(255 * alpha), int(255 * alpha))
        cv2.line(img, self.p1, self.p2, c_inner, max(1, self.thickness - 1), cv2.LINE_AA)
        return True


class BloodEmber:
    """Rising demonic ash & cursed blood ember."""
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.reset()

    def reset(self):
        self.x = random.uniform(0, self.w)
        self.y = random.uniform(self.h * 0.7, self.h + 20)
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = random.uniform(-2.5, -6.0)
        self.size = random.randint(2, 5)
        self.life = random.randint(30, 60)
        self.max_life = self.life
        self.color = random.choice([(30, 40, 255), (20, 120, 255), (50, 50, 240), (255, 200, 255)])

    def update_and_draw(self, img):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        alpha = max(0.0, self.life / self.max_life)

        if self.life <= 0 or self.y < 10:
            self.reset()
            return

        b, g, r = self.color
        c = (int(b * alpha), int(g * alpha), int(r * alpha))
        cv2.circle(img, (int(self.x), int(self.y)), self.size, c, -1, cv2.LINE_AA)


class GlassShard:
    """Obsidian barrier shattered fragments."""
    def __init__(self, cx, cy, is_sukuna=False):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(8.0, 22.0)
        self.x = float(cx)
        self.y = float(cy)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.size = random.uniform(10, 28)
        self.angle = random.uniform(0, 360)
        self.rot_speed = random.uniform(-12, 12)
        self.life = random.randint(25, 45)
        self.max_life = self.life
        if is_sukuna:
            self.color = random.choice([(30, 40, 255), (20, 100, 255), (255, 255, 255)])
        else:
            self.color = random.choice([(255, 240, 120), (255, 120, 255), (100, 255, 255), (255, 255, 255)])

    def update_and_draw(self, img):
        self.x += self.vx
        self.y += self.vy
        self.angle += self.rot_speed
        self.life -= 1
        alpha = max(0.0, self.life / self.max_life)

        if self.life <= 0:
            return False

        rad = math.radians(self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        s = self.size * alpha

        pts = np.array([
            [self.x - s * cos_a, self.y - s * sin_a],
            [self.x + s * sin_a * 0.6, self.y - s * cos_a * 0.6],
            [self.x + s * cos_a * 0.8, self.y + s * sin_a * 0.8],
            [self.x - s * sin_a * 0.5, self.y + s * cos_a * 0.5]
        ], dtype=np.int32)

        b, g, r = self.color
        c = (int(b * alpha), int(g * alpha), int(r * alpha))
        cv2.polylines(img, [pts], isClosed=True, color=c, thickness=2, lineType=cv2.LINE_AA)
        return True


# -----------------------------------------------------------------------------
# DUAL DOMAIN EXPANSION STATE MANAGER
# -----------------------------------------------------------------------------
class DualDomainEngine:
    def __init__(self, width=1280, height=720):
        self.w = width
        self.h = height
        self.cx = width // 2
        self.cy = height // 2

        self.sound = DualDomainSoundEngine()
        self.hand_filter = OneEuroPointFilter(min_cutoff=1.1, beta=0.04)
        
        self.left_lm_smoother = HandLandmarkSmoother(min_cutoff=1.0, beta=0.02)
        self.right_lm_smoother = HandLandmarkSmoother(min_cutoff=1.0, beta=0.02)

        # Domain State: "IDLE", "CHARGING", "EXPANDING", "ACTIVE", "SHATTERING"
        self.state = "IDLE"
        self.domain_type = None  # "GOJO" or "SUKUNA"
        self.charge_progress = 0.0
        self.barrier_radius = 0.0

        # Gojo: Space & Stars
        self.bg_stars = [BackgroundStar(width, height) for _ in range(320)]
        self.warp_stars = [WarpStar(width, height) for _ in range(180)]
        self.shooting_stars = [ShootingStar(width, height) for _ in range(3)]

        # Sukuna: Slashes & Blood Embers
        self.slashes = []
        self.embers = [BloodEmber(width, height) for _ in range(40)]

        # FX
        self.shards = []
        self.shake_intensity = 0.0
        self.whiteout_alpha = 0.0
        self.hand_anchor = None

    def trigger_screen_shake(self, intensity):
        self.shake_intensity = max(self.shake_intensity, intensity)

    def trigger_whiteout(self, alpha=1.0):
        self.whiteout_alpha = max(self.whiteout_alpha, alpha)

    def activate_domain(self, domain_type, anchor):
        if self.state in ["IDLE", "CHARGING"]:
            self.domain_type = domain_type
            self.state = "EXPANDING"
            self.barrier_radius = 10.0
            self.hand_anchor = anchor
            if domain_type == "GOJO":
                self.sound.play("gojo_expand")
            else:
                self.sound.play("sukuna_expand")
            self.trigger_screen_shake(26)
            self.trigger_whiteout(0.9)

    def shatter_domain(self):
        if self.state in ["ACTIVE", "EXPANDING"]:
            is_sukuna = (self.domain_type == "SUKUNA")
            self.state = "SHATTERING"
            self.sound.play("shatter")
            self.trigger_screen_shake(20)
            self.shards = [GlassShard(self.cx, self.cy, is_sukuna=is_sukuna) for _ in range(65)]
            self.charge_progress = 0.0
            self.barrier_radius = 0.0

    def render_malevolent_shrine(self, img, frame_webcam):
        """Renders Sukuna's Malevolent Shrine (Fukuma Mizushi) detailed temple architecture."""
        h, w = img.shape[:2]
        cx, cy = self.cx, self.cy

        # 1. Dark blood-red cursed hellscape background
        crimson_base = np.zeros_like(img)
        crimson_base[:, :] = (15, 6, 32)  # Deep blood red
        cv2.addWeighted(frame_webcam, 0.22, crimson_base, 0.78, 0, img)

        # Palette
        c_wood_dark = (14, 8, 32)        # Deep obsidian wood
        c_wood_mid = (22, 16, 58)        # Dark crimson mahogany
        c_roof_tile = (10, 6, 26)        # Charcoal-crimson roof tiles
        c_gold_trim = (35, 75, 200)      # Ominous gold/amber edge
        c_crimson_glow = (20, 30, 255)   # Blood red glow

        # 2. Base altar / foundation steps
        base_pts = np.array([
            [cx - 440, cy + 300],
            [cx + 440, cy + 300],
            [cx + 360, cy + 200],
            [cx - 360, cy + 200]
        ], np.int32)
        cv2.fillPoly(img, [base_pts], c_wood_dark)
        cv2.polylines(img, [base_pts], True, c_gold_trim, 2, cv2.LINE_AA)

        # Foundation step horizontal lines
        for step_y in [cy + 230, cy + 260]:
            cv2.line(img, (cx - 380, step_y), (cx + 380, step_y), (25, 18, 50), 2, cv2.LINE_AA)

        # 3. Four Heavy Temple Columns / Pillars
        pillar_xs = [cx - 270, cx - 95, cx + 95, cx + 270]
        for px in pillar_xs:
            cv2.rectangle(img, (px - 15, cy + 15), (px + 15, cy + 200), c_wood_mid, -1)
            cv2.rectangle(img, (px - 15, cy + 15), (px + 15, cy + 200), c_gold_trim, 2)

        # 4. Central Demonic Skull Sanctum / Inner Cavern
        cv2.rectangle(img, (cx - 80, cy + 50), (cx + 80, cy + 200), (6, 3, 15), -1)

        # Gaping demonic skull mouth with fangs
        cv2.ellipse(img, (cx, cy + 135), (58, 38), 0, 0, 360, (4, 2, 10), -1, cv2.LINE_AA)
        for i in range(-3, 4):
            fx = cx + i * 15
            # Upper razor fangs
            cv2.line(img, (fx, cy + 110), (fx, cy + 128), (210, 210, 245), 2, cv2.LINE_AA)
            # Lower razor fangs
            cv2.line(img, (fx, cy + 160), (fx, cy + 144), (210, 210, 245), 2, cv2.LINE_AA)

        # Glowing sinister red demon eyes inside sanctum
        blend_glow(img, (cx - 32, cy + 85), 28, c_crimson_glow)
        blend_glow(img, (cx + 32, cy + 85), 28, c_crimson_glow)
        cv2.circle(img, (cx - 32, cy + 85), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (cx + 32, cy + 85), 5, (255, 255, 255), -1, cv2.LINE_AA)

        # 5. Lower Pagoda Roof Tier (Curved Japanese Temple Eaves)
        lower_roof = np.array([
            [cx - 400, cy + 25],
            [cx - 250, cy - 25],
            [cx, cy - 40],
            [cx + 250, cy - 25],
            [cx + 400, cy + 25],
            [cx + 350, cy + 42],
            [cx, cy - 18],
            [cx - 350, cy + 42]
        ], np.int32)
        cv2.fillPoly(img, [lower_roof], c_roof_tile)
        cv2.polylines(img, [lower_roof], True, c_gold_trim, 2, cv2.LINE_AA)

        # 6. Upper Grand Pagoda Roof (Majestic Flared Gables)
        upper_roof = np.array([
            [cx - 330, cy - 60],
            [cx - 170, cy - 140],
            [cx, cy - 175],
            [cx + 170, cy - 140],
            [cx + 330, cy - 60],
            [cx + 290, cy - 45],
            [cx, cy - 150],
            [cx - 290, cy - 45]
        ], np.int32)
        cv2.fillPoly(img, [upper_roof], c_roof_tile)
        cv2.polylines(img, [upper_roof], True, c_gold_trim, 3, cv2.LINE_AA)

        # 7. Giant Demonic Horns sweeping out of the temple top
        horn_l = np.array([
            [cx - 85, cy - 175],
            [cx - 190, cy - 275],
            [cx - 245, cy - 305],
            [cx - 180, cy - 250],
            [cx - 65, cy - 175]
        ], np.int32)
        cv2.fillPoly(img, [horn_l], (18, 12, 45))
        cv2.polylines(img, [horn_l], True, c_gold_trim, 2, cv2.LINE_AA)

        horn_r = np.array([
            [cx + 85, cy - 175],
            [cx + 190, cy - 275],
            [cx + 245, cy - 305],
            [cx + 180, cy - 250],
            [cx + 65, cy - 175]
        ], np.int32)
        cv2.fillPoly(img, [horn_r], (18, 12, 45))
        cv2.polylines(img, [horn_r], True, c_gold_trim, 2, cv2.LINE_AA)

        # 8. Hanging Red Torii / Temple Lanterns
        lantern_positions = [
            (cx - 390, cy + 48), (cx + 390, cy + 48),
            (cx - 320, cy - 35), (cx + 320, cy - 35)
        ]
        for lx, ly in lantern_positions:
            blend_glow(img, (lx, ly), 32, (20, 40, 255))
            cv2.circle(img, (lx, ly), 9, (25, 45, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (lx, ly), 13, (40, 160, 255), 2, cv2.LINE_AA)

        # 9. Torrential Flurry of Cleave & Dismantle Slashes
        if random.random() < 0.90:
            self.slashes.append(DismantleSlash(w, h))
            if random.random() < 0.55:
                self.slashes.append(DismantleSlash(w, h))

        self.slashes = [sl for sl in self.slashes if sl.update_and_draw(img)]

        # 10. Rising Cursed Blood Embers
        for em in self.embers:
            em.update_and_draw(img)

        # 11. Violent Crimson Lightning Strikes
        if random.random() < 0.45:
            x1 = random.randint(0, w)
            x2 = random.randint(0, w)
            draw_lightning(img, (x1, 0), (x2, random.randint(cy - 100, h)), (30, 40, 255), thickness=2, segments=6, jitter=22)

    def render_infinite_void(self, img, frame_webcam):
        """Renders Gojo's pure deep space and stars."""
        t = time.time()
        space_base = np.zeros_like(img)
        space_base[:, :] = (12, 5, 8)
        cv2.addWeighted(frame_webcam, 0.20, space_base, 0.80, 0, img)

        for star in self.bg_stars:
            star.draw(img, t)

        for star in self.warp_stars:
            star.update_and_draw(img, self.cx, self.cy, speed=0.18)

        for s in self.shooting_stars:
            s.update_and_draw(img, 0.033)

    def update_and_render(self, img, frame_webcam):
        h, w = img.shape[:2]

        if self.state == "CHARGING":
            if self.hand_anchor:
                ax, ay = self.hand_anchor
                r = int(12 + self.charge_progress * 48)
                if self.domain_type == "GOJO":
                    blend_glow(img, (ax, ay), r * 2, (255, 200, 50))
                    blend_glow(img, (ax, ay), r, (255, 100, 255))
                    col_light = (255, 220, 80)
                else:
                    blend_glow(img, (ax, ay), r * 2, (20, 30, 255))
                    blend_glow(img, (ax, ay), r, (40, 100, 255))
                    col_light = (40, 50, 255)

                cv2.circle(img, (ax, ay), max(3, int(r * 0.3)), (255, 255, 255), -1, cv2.LINE_AA)
                for _ in range(2):
                    lx = ax + random.randint(-r, r)
                    ly = ay + random.randint(-r, r)
                    draw_lightning(img, (ax, ay), (lx, ly), col_light, 2, 4, 12)

        elif self.state == "EXPANDING":
            self.barrier_radius += 45.0
            max_dist = math.hypot(w, h)

            if self.barrier_radius >= max_dist:
                self.state = "ACTIVE"
                self.trigger_screen_shake(16)
            else:
                ax, ay = self.hand_anchor if self.hand_anchor else (self.cx, self.cy)
                br = int(self.barrier_radius)
                c_edge = (255, 220, 80) if self.domain_type == "GOJO" else (30, 40, 255)
                c_inner = (240, 60, 220) if self.domain_type == "GOJO" else (40, 100, 255)
                cv2.circle(img, (ax, ay), br, c_edge, 6, cv2.LINE_AA)
                cv2.circle(img, (ax, ay), int(br * 0.96), c_inner, 4, cv2.LINE_AA)

        elif self.state == "ACTIVE":
            if self.domain_type == "GOJO":
                self.render_infinite_void(img, frame_webcam)
            elif self.domain_type == "SUKUNA":
                self.render_malevolent_shrine(img, frame_webcam)

        elif self.state == "SHATTERING":
            surviving = [shard for shard in self.shards if shard.update_and_draw(img)]
            self.shards = surviving
            if not self.shards:
                self.state = "IDLE"
                self.domain_type = None

        # Screen Shake
        if self.shake_intensity > 0.5:
            dx = random.uniform(-self.shake_intensity, self.shake_intensity)
            dy = random.uniform(-self.shake_intensity, self.shake_intensity)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            img[:, :] = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            self.shake_intensity *= 0.86

        # Whiteout Flash
        if self.whiteout_alpha > 0.005:
            white_overlay = np.full_like(img, 255)
            cv2.addWeighted(white_overlay, self.whiteout_alpha, img, 1.0 - self.whiteout_alpha, 0, img)
            self.whiteout_alpha *= 0.86


# -----------------------------------------------------------------------------
# DOMAIN MUDRA DETECTORS (GOJO & SUKUNA)
# -----------------------------------------------------------------------------
def detect_gojo_mudra(landmarks, width, height):
    """
    Gojo's Single-Hand Mudra:
    - Index & Middle extended with fingertips crossed / touching intimately
    - Ring & Pinky curled tightly down
    """
    wrist = landmarks[0]
    d_idx_tip = math.hypot(landmarks[8].x - wrist.x, landmarks[8].y - wrist.y)
    d_idx_pip = math.hypot(landmarks[6].x - wrist.x, landmarks[6].y - wrist.y)
    index_ext = d_idx_tip > d_idx_pip * 1.10

    d_mid_tip = math.hypot(landmarks[12].x - wrist.x, landmarks[12].y - wrist.y)
    d_mid_pip = math.hypot(landmarks[10].x - wrist.x, landmarks[10].y - wrist.y)
    middle_ext = d_mid_tip > d_mid_pip * 1.10

    d_ring_tip = math.hypot(landmarks[16].x - wrist.x, landmarks[16].y - wrist.y)
    d_ring_pip = math.hypot(landmarks[14].x - wrist.x, landmarks[14].y - wrist.y)
    ring_curled = d_ring_tip < d_ring_pip * 1.25

    d_pinky_tip = math.hypot(landmarks[20].x - wrist.x, landmarks[20].y - wrist.y)
    d_pinky_pip = math.hypot(landmarks[18].x - wrist.x, landmarks[18].y - wrist.y)
    pinky_curled = d_pinky_tip < d_pinky_pip * 1.25

    tip_dist_px = math.hypot((landmarks[8].x - landmarks[12].x) * width,
                             (landmarks[8].y - landmarks[12].y) * height)
    tips_crossed = tip_dist_px < 36.0
    is_split = tip_dist_px >= 60.0

    mid_anchor = (int((landmarks[8].x + landmarks[12].x) * 0.5 * width),
                  int((landmarks[8].y + landmarks[12].y) * 0.5 * height))

    is_mudra = index_ext and middle_ext and ring_curled and pinky_curled and not is_split
    return is_mudra, mid_anchor, is_split


def detect_sukuna_mudra(lm_left, lm_right, width, height):
    """
    Sukuna's Two-Hand Clasped Stance (Enma-ten Mudra):
    - Both hands brought together at center
    - Wrists close (< 380px)
    - Index fingertips touching at top (< 85px) pointing upwards
    - Middle fingertips touching / close (< 110px)
    """
    w_l = (int(lm_left[0].x * width), int(lm_left[0].y * height))
    w_r = (int(lm_right[0].x * width), int(lm_right[0].y * height))
    wrist_dist = math.hypot(w_l[0] - w_r[0], w_l[1] - w_r[1])

    idx_l = (int(lm_left[8].x * width), int(lm_left[8].y * height))
    idx_r = (int(lm_right[8].x * width), int(lm_right[8].y * height))
    idx_dist = math.hypot(idx_l[0] - idx_r[0], idx_l[1] - idx_r[1])

    mid_l = (int(lm_left[12].x * width), int(lm_left[12].y * height))
    mid_r = (int(lm_right[12].x * width), int(lm_right[12].y * height))
    mid_dist = math.hypot(mid_l[0] - mid_r[0], mid_l[1] - mid_r[1])

    idx_l_up = lm_left[8].y < lm_left[6].y
    idx_r_up = lm_right[8].y < lm_right[6].y

    is_split = wrist_dist >= 450 or idx_dist >= 150
    is_active = (wrist_dist < 380 and idx_dist < 85 and mid_dist < 110 and idx_l_up and idx_r_up) and not is_split
    center = (int((idx_l[0] + idx_r[0]) / 2), int((idx_l[1] + idx_r[1]) / 2))
    return is_active, center, is_split


def draw_telemetry_hud(img, fps, statuses):
    """Render developer HUD overlay showing live FPS and real-time domain telemetry."""
    if not statuses and fps <= 0:
        return
    h_box = 28 + len(statuses) * 22
    overlay = img.copy()
    cv2.rectangle(overlay, (15, 15), (460, 15 + h_box), (10, 10, 15), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.rectangle(img, (15, 15), (460, 15 + h_box), (100, 240, 120), 1, cv2.LINE_AA)

    # Render FPS
    cv2.putText(img, f"FPS: {int(fps)}", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 140), 2, cv2.LINE_AA)

    # Render movement & gesture state tags
    for i, st in enumerate(statuses):
        cv2.putText(img, st, (25, 62 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA)


def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.45,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    print("==================================================================")
    print("  DUAL DOMAIN EXPANSION: INFINITE VOID & MALEVOLENT SHRINE        ")
    print("==================================================================")
    print("  DOMAINS:")
    print("  1. Satoru Gojo: INFINITE VOID (Space & Stars)")
    print("     - Gesture: 1 Hand crossed index & middle fingers")
    print("  2. Ryomen Sukuna: MALEVOLENT SHRINE (Cleave & Dismantle Hellscape)")
    print("     - Gesture: 2 Hands clasped together with index fingers up")
    print("  CONTROLS:")
    print("  - Hold gesture to expand domain! Release to shatter barrier.")
    print("  - [1] Trigger Infinite Void    [2] Trigger Malevolent Shrine")
    print("  - [SPACE] Toggle / Shatter     [M] Mirror Camera    [Q/ESC] Quit")
    print("==================================================================")

    engine = DualDomainEngine(1280, 720)
    is_mirrored = True

    prev_time = time.time()
    fps_smooth = 30.0

    with vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                time.sleep(0.01)
                continue

            if is_mirrored:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            curr_time = time.time()

            # Calculate FPS
            dt = max(1e-4, curr_time - prev_time)
            prev_time = curr_time
            fps_smooth = 0.90 * fps_smooth + 0.10 * (1.0 / dt)

            raw_frame_copy = frame.copy()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(curr_time * 1000)
            results = detector.detect_for_video(mp_image, timestamp_ms)

            num_detected = len(results.hand_landmarks) if results.hand_landmarks else 0

            detected_domain = None
            detected_anchor = None
            is_domain_split = False
            status_tags = []

            # Render hand skeletons & detect mudras
            if results.hand_landmarks:
                if len(results.hand_landmarks) == 2:
                    h_sorted = sorted(results.hand_landmarks, key=lambda lm: lm[0].x)
                    lm_l, lm_r = h_sorted[0], h_sorted[1]
                    
                    lm_l = engine.left_lm_smoother.update(lm_l)
                    lm_r = engine.right_lm_smoother.update(lm_r)
                    
                    draw_hand_skeleton(frame, lm_l, w, h, domain_type=engine.domain_type if engine.state == "ACTIVE" else None)
                    draw_hand_skeleton(frame, lm_r, w, h, domain_type=engine.domain_type if engine.state == "ACTIVE" else None)

                    # Gojo can be triggered by either hand
                    is_g1, g_anc1, g_split1 = detect_gojo_mudra(lm_l, w, h)
                    is_g2, g_anc2, g_split2 = detect_gojo_mudra(lm_r, w, h)
                    
                    if is_g1 or is_g2:
                        detected_domain = "GOJO"
                        detected_anchor = g_anc1 if is_g1 else g_anc2
                        status_tags.append("Gojo : Fingers Touched & Crossed")
                    if (g_split1 or g_split2) and engine.domain_type == "GOJO":
                        is_domain_split = True
                        
                    # Check Sukuna
                    is_s, s_anc, s_split = detect_sukuna_mudra(lm_l, lm_r, w, h)
                    if is_s:
                        detected_domain = "SUKUNA"
                        detected_anchor = s_anc
                        status_tags.append("Sukuna : Two Hands (Fingers Touched)")
                    if s_split and engine.domain_type == "SUKUNA":
                        is_domain_split = True
                        
                elif len(results.hand_landmarks) == 1:
                    landmarks = results.hand_landmarks[0]
                    
                    raw_hand = "Right"
                    if results.handedness:
                        cat = results.handedness[0][0]
                        raw_hand = cat.category_name or cat.display_name or "Right"

                    hand_label = ("Left" if raw_hand == "Right" else "Right") if is_mirrored else raw_hand
                    
                    if hand_label == "Left":
                        landmarks = engine.left_lm_smoother.update(landmarks)
                        engine.right_lm_smoother.reset()
                    else:
                        landmarks = engine.right_lm_smoother.update(landmarks)
                        engine.left_lm_smoother.reset()
                        
                    draw_hand_skeleton(frame, landmarks, w, h, domain_type=engine.domain_type if engine.state == "ACTIVE" else None)
                    
                    is_g, g_anc, g_split = detect_gojo_mudra(landmarks, w, h)
                    if is_g:
                        detected_domain = "GOJO"
                        detected_anchor = g_anc
                        status_tags.append("Gojo : Fingers Touched & Crossed")
                    if g_split and engine.domain_type == "GOJO":
                        is_domain_split = True
            else:
                engine.left_lm_smoother.reset()
                engine.right_lm_smoother.reset()

            # Domain Expansion Logic
            if detected_domain is not None:
                smoothed_anchor = engine.hand_filter.update(detected_anchor)
                if engine.state == "IDLE":
                    engine.state = "CHARGING"
                    engine.domain_type = detected_domain
                    engine.charge_progress = 0.05
                    engine.hand_anchor = smoothed_anchor
                    status_tags.append(f"Charging {detected_domain} Domain...")
                elif engine.state == "CHARGING":
                    engine.hand_anchor = smoothed_anchor
                    engine.domain_type = detected_domain
                    engine.charge_progress = min(1.0, engine.charge_progress + 0.035)
                    status_tags.append(f"Charge: {int(engine.charge_progress * 100)}% -> Expanding!")
                    if engine.charge_progress >= 0.95:
                        engine.activate_domain(detected_domain, smoothed_anchor)
                elif engine.state == "ACTIVE":
                    engine.hand_anchor = smoothed_anchor
            else:
                if is_domain_split:
                    engine.hand_filter.reset()
                    if engine.state == "CHARGING":
                        engine.state = "IDLE"
                        engine.charge_progress = 0.0
                        engine.domain_type = None
                    elif engine.state == "ACTIVE":
                        engine.shatter_domain()
                else:
                    if engine.state in ["ACTIVE", "CHARGING"]:
                        status_tags.append(f"{engine.domain_type} Maintained (Mudra Imprecise)")

            # Render Active Domain Environment
            engine.update_and_render(frame, raw_frame_copy)

            # Display Domain Title Header & Status tags
            if engine.state == "ACTIVE":
                if engine.domain_type == "GOJO":
                    status_tags.append("Domain Status: Infinite Void Active ")
                    cv2.putText(frame, "DOMAIN EXPANSION: INFINITE VOID", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 240, 100), 2, cv2.LINE_AA)
                elif engine.domain_type == "SUKUNA":
                    status_tags.append("Domain Status: Malevolent Shrine Active ")
                    cv2.putText(frame, "DOMAIN EXPANSION: MALEVOLENT SHRINE", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.85, (30, 40, 255), 2, cv2.LINE_AA)
            elif engine.state == "SHATTERING":
                status_tags.append("Domain Status: Shattering Barrier!")
            elif not status_tags:
                status_tags.append("Status: Waiting for Domain .")

            # Render Developer FPS & Gesture Telemetry HUD
            draw_telemetry_hud(frame, fps_smooth, status_tags)

            cv2.imshow("Dual Domain Expansion", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('1'):  # Force Gojo's Infinite Void
                if engine.state == "ACTIVE":
                    engine.shatter_domain()
                else:
                    engine.activate_domain("GOJO", (w // 2, h // 2))
            elif key == ord('2'):  # Force Sukuna's Malevolent Shrine
                if engine.state == "ACTIVE":
                    engine.shatter_domain()
                else:
                    engine.activate_domain("SUKUNA", (w // 2, h // 2))
            elif key == 32:  # SPACE key toggle
                if engine.state == "ACTIVE":
                    engine.shatter_domain()
                else:
                    engine.activate_domain("GOJO", (w // 2, h // 2))
            elif key == ord('m') or key == ord('M'):
                is_mirrored = not is_mirrored

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
