# EVE Benchmark for GLAMIA

This directory contains the reproducible benchmark evaluation suite for the **GLAMIA** framework on the **EVE dataset** using the frontal center webcam (`webcam_c.mp4`) and gold-standard Tobii Pro Spectrum eye-tracker ground truth (`webcam_c.h5`).

---

## 1. Research Context & Domain Alignment

### Experimental Apparatus & Setup
The EVE dataset provides continuous 30 FPS video recordings under realistic screen interaction on a **25-inch 1080p desktop display** ($1920 \times 1080$, physical diagonal $\approx 63.5\text{ cm}$):
- **Camera View (`webcam_c.mp4`)**: Frontal center webcam mounted directly above the monitor at a viewing distance of $60\text{--}70\text{ cm}$, matching standard desktop and laptop configurations.
- **Gold-Standard Ground Truth (`webcam_c.h5`)**: Continuous Point-of-Gaze pixel coordinates (`face_PoG_tobii`) captured at 1200 Hz by an infrared Tobii Pro Spectrum eye tracker ($0.288\text{ mm/pixel}$).
- **Three Stimulus Modalities**:
  1. **Image Viewing**: Static photograph exploration (~60 images, 3 seconds each).
  2. **Video Watching**: Dynamic movie trailers, animations, and sports (~12 minutes).
  3. **Wikipedia Reading**: Natural article reading (3 $\times$ 2-minute sessions).

### Geometric Envelope Fit
On a 25-inch monitor viewed at 65 cm, the display subtends an angular cone of approximately $\pm 18^\circ$ horizontally and $\pm 11^\circ$ vertically. This viewing cone lies entirely within GLAMIA's mathematically validated safe linear operating envelope, allowing the 2D linear mapper to operate with minimal perspective distortion (in theory).

---

## 2. Calibration Methodology (Few-Shot Personalization)

Because EVE does not include a dedicated interactive 9-point calibration routine or video at the start of recordings:
1. **Grid Anchor Selection**: We define the standard $3 \times 3$ calibration grid across the $1920 \times 1080$ display, matching the calibration procedure described in the paper.
2. **Opportunistic Sampling**: For each participant, we identify the **9 frames whose Tobii ground-truth gaze coordinates $(x_{\text{gt}}, y_{\text{gt}})$ are closest to these target grid positions**.
3. **Model Fitting**: GLAMIA's 2D linear mapper (Ordinary Least Squares) is trained once per participant on these 9 samples:
   $$\hat{x} = w_{x,\text{pitch}} \cdot \theta_{\text{pitch}} + w_{x,\text{yaw}} \cdot \theta_{\text{yaw}} + b_x$$
   $$\hat{y} = w_{y,\text{pitch}} \cdot \theta_{\text{pitch}} + w_{y,\text{yaw}} \cdot \theta_{\text{yaw}} + b_y$$
4. **Held-out Evaluation**: The fitted mapper is evaluated across all remaining $(N - 9)$ held-out frames ($\sim 32{,}400$ frames per validation participant, and $\sim 1.49$ million frames across the full cohort).

---

## 3. Evaluation Splits & Academic Precedents

### A. Validation Split (`val01`–`val05`, 5 Unseen Participants) — Direct Literature Baseline
- **Scale & Frame Count**: **169,860 total frames** (**162,260 valid Tobii ground-truth test frames** across 5 participants).
- **Purpose**: Direct, apples-to-apples comparison against published baselines on EVE (**EyeNet** and **EFE**).
- **Why Validation Split?**: In the official EVE release, competition test set labels (`test01`–`test10`) are withheld by the authors for blind 0-shot server challenges. Evaluating few-shot personalized models (which require ground truth for the 9 calibration targets) necessitates an unblinded split with verifiable Point-of-Gaze labels.
- **Precedent**: The EVE authors themselves (*Park et al., ECCV 2020*) evaluated their **9-point SVR personalized baseline ($172.70\text{ px}$)** on this exact validation split.

### B. Extended Population Split (`train01`–`train39` + `val01`–`val05`, 44 Unseen Participants)
- **Scale & Frame Count**: **1,442,490 total frames** in the train split (**1,334,599 valid Tobii test frames** across 39 participants), yielding a combined cohort of **1,612,350 total frames** (**1,496,859 valid Tobii test frames** across all 44 participants).
- **Purpose**: Large-scale population generalization study across **1.49 million continuous frames**.
- **Zero Data Leakage**: Because GLAMIA's 3D backbone (MobileOne-S1) was trained strictly on **Gaze360** and was never exposed to EVE data, all 44 participants represent 100% unseen test subjects.
---
## 4. Benchmark Execution

### Step 1: 3D Feature Extraction (Run Once)

Extracts continuous 3D gaze angles (`pitch`, `yaw`) from `webcam_c.mp4` and pairs each frame with Tobii `face_PoG_tobii`:

```bash
# Run from my-gaze-model/inference

# 1. Extract Validation Split (val01-val05, ~162k frames)
uv run 2d/benchmarks/EVE/extract_features.py --split val

# 2. (Optional) Extract Extended Train Split (train01-train39, ~1.44M frames)
uv run 2d/benchmarks/EVE/extract_features.py --split train
```

---

### Step 2: Run Benchmark Evaluation

Evaluates global 9-point personalization and sub-task breakdown (Image vs. Video vs. Wikipedia):

```bash
# Run from my-gaze-model/inference

# 1. Evaluate Validation Split (Direct Literature Comparison against EyeNet/EFE)
uv run 2d/benchmarks/EVE/benchmark.py --split val

# 2. Evaluate Extended Train Split (39 Training Subjects)
uv run 2d/benchmarks/EVE/benchmark.py --split train

# 3. Evaluate Combined Cohort (All 44 Unseen Subjects, 1.49M frames)
uv run 2d/benchmarks/EVE/benchmark.py --split all
```

Outputs:
- Validation Split: `experiments/eve/results/` (`eve_global_benchmark.csv`, `eve_task_breakdown.csv`, `benchmark_summary.md`)
- Extended Train Split: `experiments/eve/extended/results/`
- Combined All-44 Cohort: `experiments/eve/all_44/results/`
