# ⚡ Hand Vision: Jujutsu Kaisen Cursed Techniques & Domain Expansion Engine

An advanced, real-time Computer Vision & Procedural Animation System built with **MediaPipe Tasks API**, **OpenCV**, **NumPy**, and **Sounddevice**.

This project turns your webcam into an interactive Jujutsu Kaisen cursed energy engine, featuring precise hand-gesture recognition, 1-Euro adaptive tracking, 3D perspective physics, zero-latency procedural anime sound synthesis, and immersive domain expansion visual effects.

---

## 📁 Project Architecture & Files

| File | Description | Key Features |
| :--- | :--- | :--- |
| [`hand_vision.py`](file:///c:/handvision/hand_vision.py) | **Limitless Engine** (*Blue, Red & Hollow Purple*) | Persistent floating spheres, magnetic fusion, 3D towards-camera trajectory, fullscreen whiteout impact, glowing `"HOLLOW"` typography, and anime sound effects. |
| [`hand_vision2.py`](file:///c:/handvision/hand_vision2.py) | **Dual Domain Expansion Engine** (*Infinite Void & Malevolent Shrine*) | Gojo's 1-hand crossed mudra (*Pure Space & Stars*), Sukuna's 2-hand clasped mudra (*Malevolent Shrine Pagoda & Slashes*), glass shatter physics, and live telemetry HUD. |
| `hand_landmarker.task` | **MediaPipe Hand Landmark Model** | Auto-downloaded 21-point 3D hand tracking neural network running locally on CPU/XNNPACK. |

---

## 🎮 How to Use & Gesture Guide

### 🟣 1. `hand_vision.py` — Blue, Red & Hollow Purple

Run the script:
```powershell
.\venv\Scripts\python.exe hand_vision.py
```

#### Gesture Mechanics:
1. **Left Hand — Cursed Technique Lapse: Blue (蒼 - Ao)**:
   * Form the **Two-Finger Mudra** (extend Index + Middle together, curl Ring + Pinky).
   * Blue starts as a pinpoint spark and continuously swells larger the longer you hold your hand in place.
   * **Floating Mechanism**: If you open/remove your left hand gently, **Blue detaches and remains hovering in mid-air**!
2. **Right Hand — Cursed Technique Reversal: Red (赫 - Aka)**:
   * Form the **Two-Finger Mudra** with your Right Hand.
   * Red spawns, charges, and radiates outward explosive force particles.
3. **Collision Fusion — Hollow Purple (虚式「茈」- Murasaki)**:
   * Bring your Right Hand (Red) close to the hovering/active Blue sphere.
   * A magnetic lightning arc draws them together into a collision singularity.
   * Triggers an explosive shockwave, screen shake, glowing **`HOLLOW`** banner text, and the deep anime vocalization: **`"HOLLOW..."`**!
4. **3D Towards-Camera Launch**:
   * **Flick / spread your fingers apart** to launch Blue, Red, or Hollow Purple.
   * The projectile rushes forward in 3D directly at the camera lens, growing exponentially into a colossal blast before detonating in a **blinding fullscreen whiteout flash** with hypersonic laser sound effects!

---

### ⛩️ 2. `hand_vision2.py` — Dual Domain Expansion

Run the script:
```powershell
.\venv\Scripts\python.exe hand_vision2.py
```

#### Domain 1: Satoru Gojo — **Infinite Void (無量空処 - Muryōkūsho)**
* **Hand Sign**: Single-hand **Crossed Fingers Mudra** (Index crossed over Middle finger, Ring & Pinky curled down).
* **Environment**:
  * Pure **deep space cosmos** with 320+ twinkling background stars.
  * 180+ **3D warp stars** streaming with depth perspective.
  * Graceful **shooting stars & meteors** gliding across space.
  * Ethereal **celestial bell chimes** and deep vacuum drone.

#### Domain 2: Ryomen Sukuna — **Malevolent Shrine (伏魔御廚子 - Fukuma Mizushi)**
* **Hand Sign**: Two-hand **Enma-ten Clasped Mudra** (Both hands brought together at center, wrists touching, index fingertips pointed straight up touching at the peaks).
* **Environment**:
  * Ominous dark **blood-red cursed sky**.
  * Detailed **Malevolent Shrine Temple Pagoda** architecture featuring multi-tiered sweeping eaves, 4 wooden temple pillars, demonic towering bull horns, fanged skull jaw caverns, glowing red demon eyes, and hanging temple lanterns.
  * Torrential flurry of **Cleave & Dismantle (`Hachi` & `Kai`) slashes** slicing reality with razor-white cores and crimson edges.
  * **Violent crimson lightning strikes** and rising blood embers.
  * Demonic sub-bass roar and sinister temple bell/gong audio.

#### Domain Collapse:
* Release your hand sign or press <kbd>SPACE</kbd> to shatter the domain barrier into **65+ rotating obsidian glass shards** with crystalline glass-breaking sound effects!

---

## ⌨️ Keyboard Hotkeys Reference

| Key | Action |
| :---: | :--- |
| <kbd>SPACE</kbd> | Toggle Domain Expansion on / off (in `hand_vision2.py`) |
| <kbd>1</kbd> | Force Expand **Infinite Void** (*Gojo*) |
| <kbd>2</kbd> | Force Expand **Malevolent Shrine** (*Sukuna*) |
| <kbd>C</kbd> | Clear all active spheres and shockwaves (in `hand_vision.py`) |
| <kbd>M</kbd> | Toggle Camera Mirroring on / off |
| <kbd>Q</kbd> / <kbd>ESC</kbd> | Quit Application |

---

## 🔬 Under the Hood: Mathematical & Technical Deep Dive

### 1. 1-Euro (1€) Adaptive Smoothing Filter
Standard Moving Averages introduce input lag when moving fast. The **1-Euro Filter** uses an adaptive cutoff frequency $\hat{f}_c$ based on velocity:
$$\hat{f}_c = f_{c,\min} + \beta \cdot |\hat{\dot{x}}|$$
$$\alpha = \frac{1}{1 + \frac{\tau}{\Delta t}}, \quad \tau = \frac{1}{2\pi \hat{f}_c}$$
* **When hand is stationary** ($|\dot{x}| \to 0$): Cutoff frequency drops to $f_{c,\min} \approx 1.1\,\text{Hz}$, eliminating 100% of camera sensor micro-jitter and keeping hovering spheres rock-solid.
* **When hand moves/flicks** ($|\dot{x}| \gg 0$): Cutoff frequency scales up instantly with slope $\beta = 0.04$, eliminating all drag/latency.

### 2. 3D Perspective Projection Physics ($z$-depth trajectory)
Projectiles travel along a depth axis $z \in [1.0, 0.05]$:
$$x(t) = x_{\text{hand}} + (x_{\text{screen}} - x_{\text{hand}}) \cdot (1 - z)^{0.85}$$
$$\text{Radius}(z) = R_{\text{base}} \cdot \frac{1}{\max(0.06, z)} \cdot \text{Power}$$
As $z \to 0.05$, the sphere radius scales exponentially, expanding from $40\,\text{px}$ to over $750\,\text{px}$, naturally engulfing the entire screen before triggering the whiteout flash.

### 3. Procedural Zero-Latency Audio Synthesis
Audio is synthesized purely in mathematical waveforms using **NumPy** at $44.1\,\text{kHz}$ and played asynchronously on background worker threads via `sounddevice` (0ms disk read latency):
* **"HOLLOW..." Vocal Synthesis**: Formant resonant filters at $440\,\text{Hz}$ and $840\,\text{Hz}$ with descending sub-bass pitch contour ($86\,\text{Hz} \to 64\,\text{Hz}$), harmonic sawtooth buzzing, and a $38\,\text{ms}$ stereo chorus widening delay.
* **Hypersonic Laser & Explosion**: Chirp frequency sweep ($3500\,\text{Hz} \to 55\,\text{Hz}$) + convolved low-pass noise rumble + non-linear hyperbolic tangent distortion saturation ($\tanh(1.8 \cdot x)$).
* **Domain Expansion Chimes**: Multi-octave resonant modal bells ($528\,\text{Hz}, 1056\,\text{Hz}, 1584\,\text{Hz}, 2112\,\text{Hz}, 3168\,\text{Hz}$) + sub-harmonic vacuum pulse.

### 4. Radial Glow Caching & Additive Blending
To prevent heavy CPU slowdowns from repeated Gaussian blurs on $1280 \times 720$ frames:
* High-intensity radial glow sprites are precomputed with non-linear falloff:
  $$\text{Mask}(d) = \left(\max\left(0, 1 - \frac{d}{r}\right)\right)^{1.8}$$
* Cached in a hash map `_glow_cache[(radius, color)]`.
* Blended into the target frame using fast bounding-box region clipping and SIMD-accelerated `cv2.add()`.

---

## 🛠️ Requirements & Setup

* **Python**: 3.10 – 3.14
* **Libraries**:
  * `opencv-python`
  * `mediapipe` (MediaPipe Tasks 1.0.1+)
  * `numpy`
  * `sounddevice`

To run:
```powershell
pip install opencv-python mediapipe numpy sounddevice
python hand_vision.py
python hand_vision2.py
```
