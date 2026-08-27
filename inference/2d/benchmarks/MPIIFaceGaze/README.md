# MPIIFaceGaze Benchmark for GLAMIA

This directory contains the reproducible benchmark evaluation suite for the **GLAMIA** framework on the **MPIIFaceGaze** dataset (15 participants, 37,667 in-the-wild images).

---

## 1. Research Context & Domain Alignment

### GLAMIA's Target Operational Setting
GLAMIA is designed for **continuous, single-sitting laptop interactions** (e.g., active tasks, web navigation, usability testing, or gamified tools lasting 5–30 minutes). In this setting:
- The user sits centered in front of the laptop within a standard viewing distance ($d \approx 35\text{--}60\text{ cm}$).
- Natural, unconstrained head micro-movements and posture relaxation occur continuously.
- The 2D linear mapper assumes the viewing cone remains within the mathematically proven safe linear operating envelope.

### The Nature of MPIIFaceGaze (Discontinuous & Multi-Month)
In contrast, **MPIIFaceGaze is composed of discrete, sporadic snapshots** collected in the background every 10 minutes over a **3-month unmonitored period**:
- It is **not a continuous video stream**; consecutive frames are often hours or days apart.
- Participants operated laptops under extreme unconstrained conditions (typing on laps, lying in bed, severe screen tilt angles, and off-axis head turns reaching up to $\pm 78^\circ$).
- Viewing distance ($d$) and physical seating posture vary widely across the multi-month timeline.

### Why This Experiment Then?
We evaluate on MPIIFaceGaze as an **out-of-domain stress test** to:
1. Benchmark GLAMIA's 3D feature representation on raw, unwarped public laptop webcam frames across 15 participants without requiring camera intrinsic calibration or 3D face reconstruction.
2. Quantify the exact mathematical boundary conditions and calibration decay of static 2D linear mapping when multi-day physical setup shifts occur.

---

## 2. Calibration Methodology (Emulating Few-Shot Personalization)

Because MPIIFaceGaze does not include a dedicated interactive 9-point calibration routine or video:
1. **Grid Anchor Selection**: We define a canonical $3 \times 3$ grid across the participant's display resolution (at $10\%$, $50\%$, and $90\%$ of screen width and height).
2. **Opportunistic Sampling**: We identify the **9 frames whose ground-truth on-screen coordinates $(x_{\text{gt}}, y_{\text{gt}})$ are closest to these 9 grid targets**.
3. **Model Fitting**: GLAMIA's 2D linear mapper (Ordinary Least Squares) is fitted once on these 9 samples:
   $$\hat{x} = w_{x,\text{pitch}} \cdot \theta_{\text{pitch}} + w_{x,\text{yaw}} \cdot \theta_{\text{yaw}} + b_x$$
   $$\hat{y} = w_{y,\text{pitch}} \cdot \theta_{\text{pitch}} + w_{y,\text{yaw}} \cdot \theta_{\text{yaw}} + b_y$$
4. **Held-out Evaluation**: The fitted mapper is evaluated across all remaining $(N - 9)$ held-out frames ($\sim 37{,}600$ frames total across all 15 participants).

---

## 3. Benchmark Execution

### Step 1: 3D Feature Extraction (Extract & Cache 3D Gaze Angles)

Extracts continuous 3D gaze angles (`pitch`, `yaw`) for all 37,667 frames using the pretrained MobileOne-S1 model and BlazeFace face detector:

```bash
# Run from my-gaze-model/inference
uv run 2d/benchmarks/MPIIFaceGaze/extract_features.py \
    --data-dir ../../datasets/extracted/MPIIFaceGaze \
    --weights ../../weights/prod.pth \
    --output-dir ../../experiments/mpiifacegaze \
    --device auto
```

Outputs:
- `experiments/mpiifacegaze/mpiifacegaze_features.csv`

---

### Step 2: Run Personalization & Cross-Day Benchmark

Evaluates the 9-point personalization benchmark and within-day vs. cross-day calibration decay:

```bash
# Run from my-gaze-model/inference
uv run 2d/benchmarks/MPIIFaceGaze/benchmark.py \
    --features-csv ../../experiments/mpiifacegaze/mpiifacegaze_features.csv \
    --output-dir ../../experiments/mpiifacegaze \
    --margin-ratio 0.10
```

Outputs:
- `experiments/mpiifacegaze/results/mpiifacegaze_global_benchmark.csv`
- `experiments/mpiifacegaze/results/mpiifacegaze_cross_day_decay.csv`
- `experiments/mpiifacegaze/results/benchmark_summary.md`
