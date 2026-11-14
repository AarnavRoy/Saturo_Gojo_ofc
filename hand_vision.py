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
# HIGH-FIDELITY PROCEDURAL AUDIO SYNTHESIZER
# -----------------------------------------------------------------------------
class CursedSoundEngine:
    """Zero-latency procedural sound synthesizer for anime audio effects."""
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        self.sounds = {
            "hollow": self._synth_hollow_voice(),
            "blast": self._synth_crazy_blast(),
            "fire": self._synth_fire_blast(),
            "merge": self._synth_merge_snap(),
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

    def _synth_hollow_voice(self):
        """Synthesizes a deep, resonant animated voice vocalization: 'HOLLOW...'"""
        duration = 1.5
        t = np.linspace(0, duration, int(self.sr * duration), False)

        # Deep sub-bass vocal contour (descending 86Hz -> 64Hz)
        f0 = 86.0 - 22.0 * (t / duration)
        phase = 2.0 * np.pi * np.cumsum(f0) / self.sr

        # Vocal harmonic spectrum
        buzz = (np.sin(phase) + 0.65 * np.sin(2.0 * phase) +
                0.45 * np.sin(3.0 * phase) + 0.30 * np.sin(4.0 * phase) +
                0.20 * np.sin(5.0 * phase))

        # Formants for 'H-O-L-L-O-W'
        formant1 = np.sin(2.0 * np.pi * 440.0 * t) * np.exp(-t * 2.0)
        formant2 = np.sin(2.0 * np.pi * 840.0 * t) * np.exp(-t * 1.8)
        sub_bass = np.sin(2.0 * np.pi * 48.0 * t) * (1.0 - t / duration)

        # Swelling envelope
        env = np.sin(np.pi * np.clip(t / duration, 0.0, 1.0) ** 0.6) ** 1.6
        signal = (buzz * 0.45 + formant1 * 0.30 + formant2 * 0.25 + sub_bass * 0.50) * env

        # Stereo chorus / widening delay
        delay_samples = int(self.sr * 0.038)
        chorus = np.pad(signal, (delay_samples, 0))[:-delay_samples] * 0.35
        output = signal + chorus
        return (output / (np.max(np.abs(output)) + 1e-5) * 0.90).astype(np.float32)

    def _synth_crazy_blast(self):
        """Synthesizes devastating hypersonic laser + sub-bass explosion + distortion rumble."""
        duration = 2.5
        t = np.linspace(0, duration, int(self.sr * duration), False)

        # 1. Hypersonic laser beam screech (sweep down from 3500Hz to 60Hz)
        f_laser = 3500.0 * np.exp(-t * 6.2) + 55.0
        laser = np.sin(2.0 * np.pi * np.cumsum(f_laser) / self.sr) * np.exp(-t * 3.2)

        # 2. Heavy distorted explosion rumble
        noise = np.random.uniform(-1.0, 1.0, len(t))
        kernel = np.ones(55) / 55.0
        rumble = np.convolve(noise, kernel, mode='same') * np.exp(-t * 1.5) * 2.3

        # 3. Sub-bass shockwave drop
        sub = np.sin(2.0 * np.pi * (82.0 * np.exp(-t * 2.2)) * t) * np.exp(-t * 1.3)

        # 4. Electrical crackle sparks
        crackle = (np.random.uniform(-1.0, 1.0, len(t)) > 0.93).astype(np.float32) * np.exp(-t * 3.5) * 0.55

        # Master mix with non-linear saturation
        mix = laser * 0.60 + rumble * 0.65 + sub * 0.75 + crackle * 0.40
        return (np.tanh(mix * 2.0) * 0.95).astype(np.float32)

    def _synth_fire_blast(self):
        """High-speed energy beam laser discharge."""
        duration = 0.85
        t = np.linspace(0, duration, int(self.sr * duration), False)
        f = 1900.0 * np.exp(-t * 9.0) + 70.0
        laser = np.sin(2.0 * np.pi * np.cumsum(f) / self.sr) * np.exp(-t * 4.2)
        sub = np.sin(2.0 * np.pi * 95.0 * np.exp(-t * 4.0) * t) * np.exp(-t * 2.8)
        mix = laser * 0.70 + sub * 0.60
        return (np.tanh(mix * 1.6) * 0.85).astype(np.float32)

    def _synth_merge_snap(self):
        """Electric snap and resonant chime for fusion."""
        duration = 0.65
        t = np.linspace(0, duration, int(self.sr * duration), False)
        snap = np.sin(2.0 * np.pi * 1200.0 * np.exp(-t * 18.0) * t) * np.exp(-t * 12.0)
        sub = np.sin(2.0 * np.pi * 65.0 * t) * np.exp(-t * 4.0)
        mix = snap * 0.6 + sub * 0.5
        return (mix * 0.8).astype(np.float32)


# -----------------------------------------------------------------------------
# ONE-EURO ADAPTIVE SMOOTHING FILTER (JITTER-FREE & ZERO LAG)
# -----------------------------------------------------------------------------
class OneEuroFilter:
    """1€ Filter for adaptive jitter-free point tracking."""
    def __init__(self, min_cutoff=1.2, beta=0.04, d_cutoff=1.0):
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

        # Filtered velocity
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        # Dynamic cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat


class OneEuroPointFilter:
    def __init__(self, min_cutoff=1.2, beta=0.045):
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
# GRAPHICS & GLOW PRIMITIVES
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


# Hand skeleton palette
FINGER_CONNECTIONS = {
    "palm": [(0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (0, 17)],
    "thumb": [(1, 2), (2, 3), (3, 4)],
    "index": [(5, 6), (6, 7), (7, 8)],
    "middle": [(9, 10), (10, 11), (11, 12)],
    "ring": [(13, 14), (14, 15), (15, 16)],
    "pinky": [(17, 18), (18, 19), (19, 20)],
}

FINGER_COLORS = {
    "palm": (210, 120, 255),      # Soft violet
    "thumb": (240, 50, 200),      # Magenta
    "index": (255, 235, 0),       # Cyan
    "middle": (80, 245, 120),     # Neon Green
    "ring": (30, 215, 255),       # Amber Yellow
    "pinky": (30, 130, 255),      # Orange
}

FINGERTIP_INDICES = [4, 8, 12, 16, 20]
FINGERTIP_COLORS = [
    (240, 50, 200),
    (255, 235, 0),
    (80, 245, 120),
    (30, 215, 255),
    (30, 130, 255),
]


def draw_hand_skeleton(img, landmarks, width, height):
    """Draw stylish neon skeleton and joint circles."""
    pts = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

    for finger, connections in FINGER_CONNECTIONS.items():
        color = FINGER_COLORS[finger]
        for start_idx, end_idx in connections:
            p1, p2 = pts[start_idx], pts[end_idx]
            cv2.line(img, p1, p2, color, 3, cv2.LINE_AA)

    for i, pt in enumerate(pts):
        if i not in FINGERTIP_INDICES and i != 0:
            cv2.circle(img, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, pt, 6, (60, 60, 60), 1, cv2.LINE_AA)

    cv2.circle(img, pts[0], 7, (210, 120, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pts[0], 9, (255, 255, 255), 2, cv2.LINE_AA)

    for idx, color in zip(FINGERTIP_INDICES, FINGERTIP_COLORS):
        pt = pts[idx]
        cv2.circle(img, pt, 7, color, -1, cv2.LINE_AA)
        cv2.circle(img, pt, 10, (255, 255, 255), 2, cv2.LINE_AA)


# -----------------------------------------------------------------------------
# PARTICLES & 3D TOWARDS-CAMERA PROJECTILES
# -----------------------------------------------------------------------------
class Particle:
    __slots__ = ['x', 'y', 'vx', 'vy', 'color', 'size', 'alpha', 'life', 'max_life']

    def __init__(self, x, y, vx, vy, color, size, max_life=26):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.size = size
        self.alpha = 1.0
        self.max_life = max_life
        self.life = max_life

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.alpha = max(0.0, self.life / self.max_life)
        return self.life > 0


class Shockwave:
    __slots__ = ['x', 'y', 'radius', 'max_radius', 'speed', 'color', 'thickness']

    def __init__(self, x, y, max_radius, color, speed=16, thickness=4):
        self.x = int(x)
        self.y = int(y)
        self.radius = 6
        self.max_radius = max_radius
        self.speed = speed
        self.color = color
        self.thickness = thickness

    def update_and_draw(self, img):
        self.radius += self.speed
        if self.radius >= self.max_radius:
            return False
        alpha = 1.0 - (self.radius / self.max_radius)
        b, g, r = self.color
        c = (int(b * alpha), int(g * alpha), int(r * alpha))
        cv2.circle(img, (self.x, self.y), int(self.radius), c, max(1, int(self.thickness * alpha)), cv2.LINE_AA)
        return True


class BlastProjectile:
    """3D Towards-Camera Projectile (Grows massively as it rushes towards the viewer)."""
    def __init__(self, p_type, start_pt, power=1.0):
        self.type = p_type  # "blue", "red", "purple"
        self.start_x = float(start_pt[0])
        self.start_y = float(start_pt[1])

        # Center of screen towards camera lens
        self.target_x = 640.0
        self.target_y = 360.0

        self.x = self.start_x
        self.y = self.start_y

        # Z-depth: 1.0 (at hand) -> 0.0 (hitting camera lens)
        self.z = 1.0
        self.z_speed = 0.038 if p_type == "purple" else 0.052

        self.power = power
        self.base_radius = 52.0 if p_type == "purple" else 36.0
        self.radius = self.base_radius
        self.trail = []

    def update(self, engine):
        self.z -= self.z_speed

        # Approach screen center smoothly
        t = 1.0 - max(0.0, self.z)
        self.x = self.start_x + (self.target_x - self.start_x) * (t * 0.45)
        self.y = self.start_y + (self.target_y - self.start_y) * (t * 0.45)

        # 3D perspective growth: rapidly expands as it nears the screen
        scale = 1.0 / max(0.06, self.z)
        self.radius = self.base_radius * scale * self.power

        self.trail.append((int(self.x), int(self.y), int(self.radius * 0.4)))
        if len(self.trail) > 10:
            self.trail.pop(0)

        # Screen shake builds up as blast rushes closer
        shake = (1.0 - self.z) * (26.0 if self.type == "purple" else 15.0)
        engine.trigger_screen_shake(shake)

        # IMPACT / WHITEOUT TRIGGER
        if self.z <= 0.06:
            if self.type == "purple":
                engine.trigger_whiteout(1.0)  # Pure blinding whiteout flash!
                engine.shockwaves.append(Shockwave(int(self.x), int(self.y), 700, (255, 80, 255), speed=32, thickness=9))
                engine.trigger_screen_shake(38)
            elif self.type == "blue":
                engine.trigger_whiteout(0.85)
                engine.shockwaves.append(Shockwave(int(self.x), int(self.y), 500, (255, 200, 40), speed=26, thickness=7))
                engine.trigger_screen_shake(20)
            elif self.type == "red":
                engine.trigger_whiteout(0.85)
                engine.shockwaves.append(Shockwave(int(self.x), int(self.y), 500, (40, 60, 255), speed=26, thickness=7))
                engine.trigger_screen_shake(20)
            return False

        return True


class CursedAnimationEngine:
    def __init__(self):
        self.particles = []
        self.shockwaves = []
        self.projectiles = []

        # Sound Engine
        self.sound = CursedSoundEngine()

        # One-Euro Adaptive Filters for Silky Smooth Hand Tracking
        self.blue_filter = OneEuroPointFilter(min_cutoff=1.1, beta=0.04)
        self.red_filter = OneEuroPointFilter(min_cutoff=1.1, beta=0.04)
        self.purple_filter = OneEuroPointFilter(min_cutoff=1.0, beta=0.035)
        
        # Smooth entire hand gestures
        self.left_lm_smoother = HandLandmarkSmoother(min_cutoff=1.0, beta=0.02)
        self.right_lm_smoother = HandLandmarkSmoother(min_cutoff=1.0, beta=0.02)

        # Persistent Spheres State
        self.blue_sphere = None
        self.red_sphere = None
        self.purple_sphere = None

        # Screen shake & Whiteout Flash
        self.shake_intensity = 0.0
        self.whiteout_alpha = 0.0

    def trigger_screen_shake(self, intensity):
        self.shake_intensity = max(self.shake_intensity, intensity)

    def trigger_whiteout(self, alpha=1.0):
        self.whiteout_alpha = max(self.whiteout_alpha, alpha)

    def render_whiteout(self, img):
        if self.whiteout_alpha > 0.005:
            white_overlay = np.full_like(img, 255)
            cv2.addWeighted(white_overlay, self.whiteout_alpha, img, 1.0 - self.whiteout_alpha, 0, img)
            self.whiteout_alpha *= 0.88

    def spawn_blue_particles(self, cx, cy, count=3, radius=42):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(radius * 0.4, radius * 1.3)
            px = cx + math.cos(angle) * dist
            py = cy + math.sin(angle) * dist
            speed = random.uniform(3.0, 6.5)
            vx = -math.cos(angle) * speed - math.sin(angle) * (speed * 1.2)
            vy = -math.sin(angle) * speed + math.cos(angle) * (speed * 1.2)
            self.particles.append(Particle(px, py, vx, vy, (255, 210, 40), random.randint(2, 5), max_life=18))

    def spawn_red_particles(self, cx, cy, count=3, radius=42):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(radius * 0.2, radius * 1.0)
            px = cx + math.cos(angle) * dist
            py = cy + math.sin(angle) * dist
            speed = random.uniform(3.5, 7.5)
            vx = math.cos(angle) * speed + random.uniform(-1, 1)
            vy = math.sin(angle) * speed + random.uniform(-1, 1)
            self.particles.append(Particle(px, py, vx, vy, (40, 100, 255), random.randint(3, 6), max_life=18))

    def spawn_purple_particles(self, cx, cy, count=6):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(15, 80)
            px = cx + math.cos(angle) * dist
            py = cy + math.sin(angle) * dist
            speed = random.uniform(3.5, 8.5)
            vx = -math.sin(angle) * speed + random.uniform(-1.5, 1.5)
            vy = math.cos(angle) * speed + random.uniform(-1.5, 1.5)
            color = random.choice([(255, 60, 200), (255, 140, 255), (210, 20, 140), (255, 255, 255)])
            self.particles.append(Particle(px, py, vx, vy, color, random.randint(3, 6), max_life=20))

    def update_and_render_particles(self, img):
        surviving = []
        for p in self.particles:
            if p.update():
                b, g, r = p.color
                c = (int(b * p.alpha), int(g * p.alpha), int(r * p.alpha))
                sz = max(1, int(p.size * p.alpha))
                cv2.circle(img, (int(p.x), int(p.y)), sz, c, -1, cv2.LINE_AA)
                surviving.append(p)
        self.particles = surviving
        self.shockwaves = [sw for sw in self.shockwaves if sw.update_and_draw(img)]

    def update_and_render_projectiles(self, img):
        surviving = []
        for proj in self.projectiles:
            if not proj.update(self):
                continue

            px, py = int(proj.x), int(proj.y)
            r = int(proj.radius)

            if proj.type == "blue":
                for tx, ty, tr in proj.trail:
                    blend_glow(img, (tx, ty), min(350, int(tr * 1.5)), (255, 120, 10))

                blend_glow(img, (px, py), min(450, int(r * 2.2)), (255, 120, 10))
                blend_glow(img, (px, py), min(280, int(r * 1.3)), (255, 220, 50))
                cv2.circle(img, (px, py), max(4, int(r * 0.35)), (255, 255, 255), -1, cv2.LINE_AA)
                self.spawn_blue_particles(px, py, count=3, radius=min(180, r))

            elif proj.type == "red":
                for tx, ty, tr in proj.trail:
                    blend_glow(img, (tx, ty), min(350, int(tr * 1.5)), (20, 30, 255))

                blend_glow(img, (px, py), min(450, int(r * 2.2)), (20, 30, 255))
                blend_glow(img, (px, py), min(280, int(r * 1.3)), (40, 140, 255))
                cv2.circle(img, (px, py), max(4, int(r * 0.35)), (255, 255, 255), -1, cv2.LINE_AA)
                self.spawn_red_particles(px, py, count=3, radius=min(180, r))

            elif proj.type == "purple":
                for tx, ty, tr in proj.trail:
                    blend_glow(img, (tx, ty), min(400, int(tr * 1.6)), (220, 30, 180))

                blend_glow(img, (px, py), min(550, int(r * 2.4)), (240, 30, 180))
                blend_glow(img, (px, py), min(360, int(r * 1.4)), (255, 160, 255))

                void_r = max(8, int(r * 0.55))
                cv2.circle(img, (px, py), void_r, (25, 0, 20), -1, cv2.LINE_AA)
                cv2.circle(img, (px, py), void_r, (255, 110, 255), max(2, int(r * 0.04)), cv2.LINE_AA)

                num_arcs = min(8, max(2, int(r / 35)))
                for _ in range(num_arcs):
                    lx = px + random.randint(-int(r * 1.1), int(r * 1.1))
                    ly = py + random.randint(-int(r * 1.1), int(r * 1.1))
                    draw_lightning(img, (px, py), (lx, ly), (255, 150, 255), thickness=max(1, int(r * 0.02)), segments=5, jitter=18)

                self.spawn_purple_particles(px, py, count=min(10, int(r / 20)))

            surviving.append(proj)

        self.projectiles = surviving


def detect_two_finger_mudra(landmarks, width, height):
    """
    Precise detection for Two-Finger Stance:
    - Index & Middle fingers extended together
    - Ring & Pinky fingers curled down
    """
    wrist = landmarks[0]

    d_idx_tip = math.hypot(landmarks[8].x - wrist.x, landmarks[8].y - wrist.y)
    d_idx_pip = math.hypot(landmarks[6].x - wrist.x, landmarks[6].y - wrist.y)
    index_ext = d_idx_tip > d_idx_pip * 1.12

    d_mid_tip = math.hypot(landmarks[12].x - wrist.x, landmarks[12].y - wrist.y)
    d_mid_pip = math.hypot(landmarks[10].x - wrist.x, landmarks[10].y - wrist.y)
    middle_ext = d_mid_tip > d_mid_pip * 1.12

    d_ring_tip = math.hypot(landmarks[16].x - wrist.x, landmarks[16].y - wrist.y)
    d_ring_pip = math.hypot(landmarks[14].x - wrist.x, landmarks[14].y - wrist.y)
    ring_curled = d_ring_tip < d_ring_pip * 1.25

    d_pinky_tip = math.hypot(landmarks[20].x - wrist.x, landmarks[20].y - wrist.y)
    d_pinky_pip = math.hypot(landmarks[18].x - wrist.x, landmarks[18].y - wrist.y)
    pinky_curled = d_pinky_tip < d_pinky_pip * 1.25

    d_tips = math.hypot(landmarks[8].x - landmarks[12].x, landmarks[8].y - landmarks[12].y)
    d_mcps = math.hypot(landmarks[5].x - landmarks[9].x, landmarks[5].y - landmarks[9].y)
    fingers_together = d_tips < d_mcps * 1.85
    is_flick_launch = d_tips >= d_mcps * 2.1 and index_ext and middle_ext

    mid_x = int((landmarks[8].x + landmarks[12].x) / 2.0 * width)
    mid_y = int((landmarks[8].y + landmarks[12].y) / 2.0 * height)

    wx = int(wrist.x * width)
    wy = int(wrist.y * height)
    aim_dx = mid_x - wx
    aim_dy = mid_y - wy

    is_split = d_tips >= d_mcps * 1.85
    is_active = index_ext and middle_ext and ring_curled and pinky_curled and not is_split
    return is_active, (mid_x, mid_y), (aim_dx, aim_dy), is_split


def render_sphere_blue(img, center, base_radius, charge, aim_vec, is_attached, engine):
    cx, cy = int(center[0]), int(center[1])
    min_r = 5
    max_r = 58
    r = int(min_r + (max_r - min_r) * charge)
    if r < 3:
        return

    blend_glow(img, (cx, cy), int(r * 2.3), (255, 120, 10))
    blend_glow(img, (cx, cy), int(r * 1.3), (255, 220, 40))
    blend_glow(img, (cx, cy), int(r * 0.7), (255, 255, 180))

    cv2.circle(img, (cx, cy), max(2, int(r * 0.35)), (255, 255, 255), -1, cv2.LINE_AA)

    p_count = max(1, int(1 + charge * 3))
    engine.spawn_blue_particles(cx, cy, count=p_count, radius=r)

    ring_r = int(r * 0.85 + math.sin(time.time() * 9) * (r * 0.2))
    cv2.circle(img, (cx, cy), max(2, ring_r), (255, 190, 50), 2, cv2.LINE_AA)

    if is_attached and aim_vec:
        v_len = math.hypot(aim_vec[0], aim_vec[1])
        if v_len > 1.0:
            lx = int(cx + (aim_vec[0] / v_len) * 320)
            ly = int(cy + (aim_vec[1] / v_len) * 320)
            cv2.line(img, (cx, cy), (lx, ly), (255, 200, 50), 1, cv2.LINE_AA)
            cv2.circle(img, (lx, ly), 4, (255, 220, 60), 1, cv2.LINE_AA)


def render_sphere_red(img, center, base_radius, charge, aim_vec, is_attached, engine):
    cx, cy = int(center[0]), int(center[1])
    min_r = 5
    max_r = 58
    r = int(min_r + (max_r - min_r) * charge)
    if r < 3:
        return

    blend_glow(img, (cx, cy), int(r * 2.3), (20, 30, 255))
    blend_glow(img, (cx, cy), int(r * 1.3), (40, 120, 255))
    blend_glow(img, (cx, cy), int(r * 0.7), (180, 230, 255))

    cv2.circle(img, (cx, cy), max(2, int(r * 0.35)), (255, 255, 255), -1, cv2.LINE_AA)

    p_count = max(1, int(1 + charge * 3))
    engine.spawn_red_particles(cx, cy, count=p_count, radius=r)

    ring_r = int(r * 0.85 + math.cos(time.time() * 9) * (r * 0.2))
    cv2.circle(img, (cx, cy), max(2, ring_r), (50, 80, 255), 2, cv2.LINE_AA)

    if is_attached and aim_vec:
        v_len = math.hypot(aim_vec[0], aim_vec[1])
        if v_len > 1.0:
            lx = int(cx + (aim_vec[0] / v_len) * 320)
            ly = int(cy + (aim_vec[1] / v_len) * 320)
            cv2.line(img, (cx, cy), (lx, ly), (40, 80, 255), 1, cv2.LINE_AA)
            cv2.circle(img, (lx, ly), 4, (50, 100, 255), 1, cv2.LINE_AA)


def render_sphere_purple(img, center, base_radius, charge, aim_vec, is_attached, engine):
    cx, cy = int(center[0]), int(center[1])
    min_r = 25
    max_r = 75
    r = int(min_r + (max_r - min_r) * charge)

    blend_glow(img, (cx, cy), int(r * 2.3), (240, 30, 180))
    blend_glow(img, (cx, cy), int(r * 1.4), (255, 140, 255))

    void_r = max(6, int(r * 0.55))
    cv2.circle(img, (cx, cy), void_r, (30, 0, 20), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), void_r, (255, 120, 255), 3, cv2.LINE_AA)

    engine.spawn_purple_particles(cx, cy, count=6)

    for _ in range(2):
        lx = cx + random.randint(-int(r * 1.2), int(r * 1.2))
        ly = cy + random.randint(-int(r * 1.2), int(r * 1.2))
        draw_lightning(img, (cx, cy), (lx, ly), (255, 150, 255), 2, 4, 10)

    # Render glowing 'HOLLOW' text on screen
    text = "HOLLOW"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.95
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    tx = cx - tw // 2
    ty = max(40, cy - r - 20)

    # Multi-layered glowing purple & bright white text
    cv2.putText(img, text, (tx, ty), font, scale, (240, 40, 180), thick + 4, cv2.LINE_AA)
    cv2.putText(img, text, (tx, ty), font, scale, (255, 160, 255), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    if is_attached and aim_vec:
        v_len = math.hypot(aim_vec[0], aim_vec[1])
        if v_len > 1.0:
            lx = int(cx + (aim_vec[0] / v_len) * 380)
            ly = int(cy + (aim_vec[1] / v_len) * 380)
            cv2.line(img, (cx, cy), (lx, ly), (255, 100, 255), 2, cv2.LINE_AA)
            cv2.circle(img, (lx, ly), 5, (255, 140, 255), 1, cv2.LINE_AA)


def draw_telemetry_hud(img, fps, statuses):
    """Render developer HUD overlay showing live FPS and real-time movement telemetry."""
    if not statuses and fps <= 0:
        return
    h_box = 28 + len(statuses) * 22
    overlay = img.copy()
    cv2.rectangle(overlay, (15, 15), (420, 15 + h_box), (10, 10, 15), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.rectangle(img, (15, 15), (420, 15 + h_box), (100, 240, 120), 1, cv2.LINE_AA)

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

    print("Hand Vision Running with 1-Euro Smooth Tracking, Audio Engine & Telemetry.")

    engine = CursedAnimationEngine()
    is_mirrored = True

    prev_time = time.time()
    fps_smooth = 30.0

    with vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                time.sleep(0.01)
                continue

            # Mirror camera horizontally
            if is_mirrored:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            curr_time = time.time()

            # Calculate FPS
            dt = max(1e-4, curr_time - prev_time)
            prev_time = curr_time
            fps_smooth = 0.90 * fps_smooth + 0.10 * (1.0 / dt)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(curr_time * 1000)
            results = detector.detect_for_video(mp_image, timestamp_ms)

            num_detected = len(results.hand_landmarks) if results.hand_landmarks else 0

            left_hand_info = None
            right_hand_info = None
            status_tags = []

            # 1. Hand skeleton rendering & tracking
            if results.hand_landmarks:
                if len(results.hand_landmarks) == 2:
                    # Sort hands left-to-right to reliably prevent swapping and instantly fusing
                    h_sorted = sorted(results.hand_landmarks, key=lambda lm: lm[0].x)
                    lm_l, lm_r = h_sorted[0], h_sorted[1]
                    
                    lm_l = engine.left_lm_smoother.update(lm_l)
                    lm_r = engine.right_lm_smoother.update(lm_r)

                    is_m_l, anc_l, vec_l, flk_l = detect_two_finger_mudra(lm_l, w, h)
                    left_hand_info = {"landmarks": lm_l, "is_mudra": is_m_l, "raw_anchor": anc_l, "aim_vec": vec_l, "is_split": flk_l, "wrist": (int(lm_l[0].x * w), int(lm_l[0].y * h))}

                    is_m_r, anc_r, vec_r, flk_r = detect_two_finger_mudra(lm_r, w, h)
                    right_hand_info = {"landmarks": lm_r, "is_mudra": is_m_r, "raw_anchor": anc_r, "aim_vec": vec_r, "is_split": flk_r, "wrist": (int(lm_r[0].x * w), int(lm_r[0].y * h))}
                    
                    draw_hand_skeleton(frame, lm_l, w, h)
                    draw_hand_skeleton(frame, lm_r, w, h)
                    
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
                        is_mudra, raw_anchor, aim_vec, is_flick = detect_two_finger_mudra(landmarks, w, h)
                        left_hand_info = {"landmarks": landmarks, "is_mudra": is_mudra, "raw_anchor": raw_anchor, "aim_vec": aim_vec, "is_split": is_flick, "wrist": (int(landmarks[0].x * w), int(landmarks[0].y * h))}
                    else:
                        landmarks = engine.right_lm_smoother.update(landmarks)
                        engine.left_lm_smoother.reset()
                        is_mudra, raw_anchor, aim_vec, is_flick = detect_two_finger_mudra(landmarks, w, h)
                        right_hand_info = {"landmarks": landmarks, "is_mudra": is_mudra, "raw_anchor": raw_anchor, "aim_vec": aim_vec, "is_split": is_flick, "wrist": (int(landmarks[0].x * w), int(landmarks[0].y * h))}
                        
                    draw_hand_skeleton(frame, landmarks, w, h)
            else:
                engine.left_lm_smoother.reset()
                engine.right_lm_smoother.reset()

            # -------------------------------------------------------------
            # 2. PURPLE EXCLUSIVE CONTROL vs BLUE/RED CREATION
            # -------------------------------------------------------------
            if engine.purple_sphere is not None:
                # HOLLOW PURPLE IS ACTIVE: NO OTHER SPHERES CAN BE CREATED!
                engine.blue_sphere = None
                engine.red_sphere = None

                if engine.purple_sphere["attached"]:
                    # Find any hand controlling Hollow Purple
                    ctrl_hand = None
                    if right_hand_info:
                        ctrl_hand = right_hand_info
                    elif left_hand_info:
                        ctrl_hand = left_hand_info

                    if ctrl_hand:
                        if ctrl_hand["is_split"]:
                            # LAUNCH HOLLOW PURPLE BLAST (TOWARDS CAMERA)!
                            status_tags.append("Flick: Hollow Purple Launched!")
                            p = engine.purple_sphere["pos"]
                            p_power = 1.0 + 0.4 * engine.purple_sphere["charge"]
                            engine.projectiles.append(BlastProjectile("purple", p, power=p_power))
                            engine.shockwaves.append(Shockwave(p[0], p[1], 230, (255, 60, 220), speed=18, thickness=6))
                            engine.trigger_screen_shake(26)
                            engine.sound.play("blast")
                            engine.purple_sphere = None
                            engine.purple_filter.reset()
                        else:
                            smoothed_pt = engine.purple_filter.update(ctrl_hand["raw_anchor"])
                            engine.purple_sphere["pos"] = list(smoothed_pt)
                            engine.purple_sphere["aim"] = ctrl_hand["aim_vec"]
                            if ctrl_hand["is_mudra"]:
                                engine.purple_sphere["charge"] = min(1.0, engine.purple_sphere["charge"] + 0.015)
                                status_tags.append("Hollow Purple Active (Fingers Touched)")
                            else:
                                status_tags.append("Holding Hollow Purple (Mudra Imprecise)")
                    else:
                        status_tags.append("Holding Hollow Purple (Hand Lost)")
                else:
                    status_tags.append("Hollow Purple Floating in Air")

            else:
                # ---------------------------------------------------------
                # 2A. BLUE SPHERE MANAGEMENT (LEFT HAND)
                # ---------------------------------------------------------
                if engine.blue_sphere is None:
                    if left_hand_info and left_hand_info["is_mudra"]:
                        status_tags.append("Left Hand: Fingers Touched (Blue Charging)")
                        smoothed_pt = engine.blue_filter.update(left_hand_info["raw_anchor"])
                        engine.blue_sphere = {
                            "pos": list(smoothed_pt),
                            "charge": 0.02,
                            "attached": True,
                            "aim": left_hand_info["aim_vec"]
                        }
                else:
                    if engine.blue_sphere["attached"]:
                        if left_hand_info:
                            if left_hand_info["is_split"]:
                                if engine.blue_sphere["charge"] > 0.15:
                                    status_tags.append("Left Hand: Flick Launch -> Blue Blast!")
                                    p = engine.blue_sphere["pos"]
                                    p_power = 0.8 + 0.5 * engine.blue_sphere["charge"]
                                    engine.projectiles.append(BlastProjectile("blue", p, power=p_power))
                                    engine.shockwaves.append(Shockwave(p[0], p[1], 140, (255, 200, 40), 18, 4))
                                    engine.trigger_screen_shake(16)
                                    engine.sound.play("fire")
                                    engine.blue_sphere = None
                                    engine.blue_filter.reset()
                                else:
                                    engine.blue_sphere["attached"] = False
                                    engine.blue_filter.reset()
                                    status_tags.append("Left Hand: Blue Floating In Mid-Air")
                            else:
                                smoothed_pt = engine.blue_filter.update(left_hand_info["raw_anchor"])
                                engine.blue_sphere["pos"] = list(smoothed_pt)
                                engine.blue_sphere["aim"] = left_hand_info["aim_vec"]
                                if left_hand_info["is_mudra"]:
                                    engine.blue_sphere["charge"] = min(1.0, engine.blue_sphere["charge"] + 0.015)
                                    status_tags.append("Left Hand: Fingers Touched (Blue Charging)")
                                else:
                                    status_tags.append("Left Hand: Holding Blue (Mudra Imprecise)")
                        else:
                            status_tags.append("Left Hand: Holding Blue (Hand Lost)")
                    else:
                        status_tags.append("Left Hand: Blue Floating In Mid-Air")

                # ---------------------------------------------------------
                # 2B. RED SPHERE MANAGEMENT (RIGHT HAND)
                # ---------------------------------------------------------
                if engine.red_sphere is None:
                    if right_hand_info and right_hand_info["is_mudra"]:
                        status_tags.append("Right Hand: Fingers Touched (Red Charging)")
                        smoothed_pt = engine.red_filter.update(right_hand_info["raw_anchor"])
                        engine.red_sphere = {
                            "pos": list(smoothed_pt),
                            "charge": 0.02,
                            "attached": True,
                            "aim": right_hand_info["aim_vec"]
                        }
                else:
                    if engine.red_sphere["attached"]:
                        if right_hand_info:
                            if right_hand_info["is_split"]:
                                if engine.red_sphere["charge"] > 0.15:
                                    status_tags.append("Right Hand: Flick Launch -> Red Blast!")
                                    p = engine.red_sphere["pos"]
                                    p_power = 0.8 + 0.5 * engine.red_sphere["charge"]
                                    engine.projectiles.append(BlastProjectile("red", p, power=p_power))
                                    engine.shockwaves.append(Shockwave(p[0], p[1], 140, (40, 60, 255), 18, 4))
                                    engine.trigger_screen_shake(16)
                                    engine.sound.play("fire")
                                    engine.red_sphere = None
                                    engine.red_filter.reset()
                                else:
                                    engine.red_sphere["attached"] = False
                                    engine.red_filter.reset()
                                    status_tags.append("Right Hand: Red Floating In Mid-Air")
                            else:
                                smoothed_pt = engine.red_filter.update(right_hand_info["raw_anchor"])
                                engine.red_sphere["pos"] = list(smoothed_pt)
                                engine.red_sphere["aim"] = right_hand_info["aim_vec"]
                                if right_hand_info["is_mudra"]:
                                    engine.red_sphere["charge"] = min(1.0, engine.red_sphere["charge"] + 0.015)
                                    status_tags.append("Right Hand: Fingers Touched (Red Charging)")
                                else:
                                    status_tags.append("Right Hand: Holding Red (Mudra Imprecise)")
                        else:
                            status_tags.append("Right Hand: Holding Red (Hand Lost)")
                    else:
                        status_tags.append("Right Hand: Red Floating In Mid-Air")

                # -----------------------------------------------------
                # CHECK MERGE WITH FLOATING BLUE SPHERE
                # -----------------------------------------------------
                if engine.blue_sphere is not None and engine.red_sphere is not None:
                    bx, by = engine.blue_sphere["pos"]
                    rx, ry = engine.red_sphere["pos"]
                    d = math.hypot(bx - rx, by - ry)

                    if d < 220:
                        status_tags.append("Magnetic Pull: Blue & Red Attracting!")
                        draw_lightning(frame, (int(bx), int(by)), (int(rx), int(ry)), (255, 100, 255), thickness=2, segments=5, jitter=16)

                        if not engine.blue_sphere["attached"]:
                            engine.blue_sphere["pos"][0] = int(0.75 * bx + 0.25 * rx)
                            engine.blue_sphere["pos"][1] = int(0.75 * by + 0.25 * ry)

                    # COLLISION MERGE!
                    if d < 65:
                        status_tags.append("COLLISION: Hollow Purple Formed!")
                        merge_x = int((bx + rx) / 2)
                        merge_y = int((by + ry) / 2)
                        comb_charge = min(1.0, max(0.5, (engine.blue_sphere["charge"] + engine.red_sphere["charge"]) / 1.5))

                        engine.purple_sphere = {
                            "pos": [merge_x, merge_y],
                            "charge": comb_charge,
                            "attached": True,
                            "aim": right_hand_info["aim_vec"] if right_hand_info else left_hand_info["aim_vec"]
                        }

                        # Play 'HOLLOW...' deep animated voice + fusion snap!
                        engine.sound.play("merge")
                        engine.sound.play("hollow")

                        engine.shockwaves.append(Shockwave(merge_x, merge_y, 180, (255, 60, 220), speed=16, thickness=5))
                        engine.trigger_screen_shake(18)
                        for _ in range(6):
                            lx = merge_x + random.randint(-120, 120)
                            ly = merge_y + random.randint(-120, 120)
                            draw_lightning(frame, (merge_x, merge_y), (lx, ly), (255, 140, 255), 2, 4, 12)

                        # BLUE AND RED CONSUMED COMPLETELY
                        engine.blue_sphere = None
                        engine.red_sphere = None

            # -------------------------------------------------------------
            # 4. RENDER ALL ACTIVE SPHERES
            # -------------------------------------------------------------
            if engine.blue_sphere is not None:
                render_sphere_blue(
                    frame,
                    engine.blue_sphere["pos"],
                    38,
                    engine.blue_sphere["charge"],
                    engine.blue_sphere["aim"],
                    engine.blue_sphere["attached"],
                    engine
                )

            if engine.red_sphere is not None:
                render_sphere_red(
                    frame,
                    engine.red_sphere["pos"],
                    38,
                    engine.red_sphere["charge"],
                    engine.red_sphere["aim"],
                    engine.red_sphere["attached"],
                    engine
                )

            if engine.purple_sphere is not None:
                render_sphere_purple(
                    frame,
                    engine.purple_sphere["pos"],
                    52,
                    engine.purple_sphere["charge"],
                    engine.purple_sphere["aim"],
                    engine.purple_sphere["attached"],
                    engine
                )

            # Update & render all particles, shockwaves, and active projectile blasts
            engine.update_and_render_particles(frame)
            engine.update_and_render_projectiles(frame)

            # Screen Shake
            if engine.shake_intensity > 0.5:
                dx = random.uniform(-engine.shake_intensity, engine.shake_intensity)
                dy = random.uniform(-engine.shake_intensity, engine.shake_intensity)
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                engine.shake_intensity *= 0.85

            # Fullscreen Impact Whiteout
            engine.render_whiteout(frame)

            if not status_tags:
                status_tags.append("Status: Waiting for Two-Finger Mudra...")

            # Render Developer FPS & Gesture Telemetry HUD
            draw_telemetry_hud(frame, fps_smooth, status_tags)

            cv2.imshow("Hand Vision", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c') or key == ord('C'):
                engine.blue_sphere = None
                engine.red_sphere = None
                engine.purple_sphere = None
                engine.projectiles.clear()
                engine.particles.clear()
                engine.shockwaves.clear()
            elif key == ord('m') or key == ord('M'):
                is_mirrored = not is_mirrored

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()