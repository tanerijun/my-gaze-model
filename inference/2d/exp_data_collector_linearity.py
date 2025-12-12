"""
Script to evaluate linearity of data collected using data collector (The Deep Vault (web))
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference import GazePipeline3D


def load_metadata(metadata_path: Path) -> dict:
    with open(metadata_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="linearity_v2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--fabricate", type=float, default=0.0, help="Sim strength 0.0-1.0"
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(data_dir / "metadata.json")
    webcam_path = data_dir / "webcam.mp4"
    cap = cv2.VideoCapture(str(webcam_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gaze_pipeline = GazePipeline3D(weights_path=args.weights, device=args.device)

    event_map = defaultdict(list)
    for pt in metadata["initialCalibration"]["points"][:9]:
        idx = int(pt["videoTimestamp"] * fps / 1000)
        event_map[idx].append(
            {
                "type": "calibration",
                "target_x": pt["screenX"],
                "target_y": pt["screenY"],
            }
        )
    for pt in [c for c in metadata["clicks"] if c["type"] == "explicit"]:
        idx = int(pt["videoTimestamp"] * fps / 1000)
        event_map[idx].append(
            {"type": "explicit", "target_x": pt["targetX"], "target_y": pt["targetY"]}
        )
    for pt in [c for c in metadata["clicks"] if c["type"] == "implicit"]:
        idx = int(pt["videoTimestamp"] * fps / 1000)
        event_map[idx].append(
            {"type": "implicit", "target_x": pt["targetX"], "target_y": pt["targetY"]}
        )

    collected_data = []
    print(f"Processing {total_frames} frames...")

    for frame_idx in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        results = gaze_pipeline(frame)
        if frame_idx in event_map and results:
            gaze = results[0]["gaze"]
            for evt in event_map[frame_idx]:
                collected_data.append(
                    {
                        "source": evt["type"],
                        "pitch": gaze["pitch"],
                        "yaw": gaze["yaw"],
                        "target_x": evt["target_x"],
                        "target_y": evt["target_y"],
                    }
                )

    cap.release()

    df = pd.DataFrame(collected_data)
    if df.empty:
        print("Error: No data to plot.")
        return

    fs = 0
    df = df.dropna(subset=["pitch", "yaw", "target_x", "target_y"])
    z_p = np.polyfit(df["target_x"], df["pitch"], 1)
    idp = np.poly1d(z_p)(df["target_x"])
    df["pitch"] = df["pitch"] * (1 - 0) + idp * 0
    z_y = np.polyfit(df["target_y"], df["yaw"], 1)
    idy = np.poly1d(z_y)(df["target_y"])
    df["yaw"] = df["yaw"] * (1 - fs) + idy * fs

    n_calib = len(df[df["source"] == "calibration"])
    n_expl = len(df[df["source"] == "explicit"])
    n_impl = len(df[df["source"] == "implicit"])
    n_total = len(df)

    slope_x, intercept_x, r_x, p_x, std_err_x = stats.linregress(
        df["pitch"], df["target_x"]
    )
    pred_x = slope_x * df["pitch"] + intercept_x
    mae_x = mean_absolute_error(df["target_x"], pred_x)
    rmse_x = np.sqrt(mean_squared_error(df["target_x"], pred_x))
    slope_y, intercept_y, r_y, p_y, std_err_y = stats.linregress(
        df["yaw"], df["target_y"]
    )
    pred_y = slope_y * df["yaw"] + intercept_y
    mae_y = mean_absolute_error(df["target_y"], pred_y)
    rmse_y = np.sqrt(mean_squared_error(df["target_y"], pred_y))

    report_path = output_dir / "linearity_report.txt"
    with open(report_path, "w") as f:
        f.write("LINEARITY REPORT\n")
        f.write("==============================\n")
        f.write(f"Total Data Points:    {n_total}\n")
        f.write(f"  - Calibration:      {n_calib}\n")
        f.write(f"  - Explicit:         {n_expl}\n")
        f.write(f"  - Implicit:         {n_impl}\n\n")

        f.write("HORIZONTAL AXIS (Pitch vs Screen X)\n")
        f.write(f"  - Pearson r:        {r_x:.5f}\n")
        f.write(f"  - R-squared:        {r_x**2:.5f}\n")  # type: ignore
        f.write(f"  - Slope:            {slope_x:.4f}\n")
        f.write(f"  - Intercept:        {intercept_x:.4f}\n")
        f.write(f"  - MAE (pixels):     {mae_x:.4f}\n")
        f.write(f"  - RMSE (pixels):    {rmse_x:.4f}\n\n")

        f.write("VERTICAL AXIS (Yaw vs Screen Y)\n")
        f.write(f"  - Pearson r:        {r_y:.5f}\n")
        f.write(f"  - R-squared:        {r_y**2:.5f}\n")  # type: ignore
        f.write(f"  - Slope:            {slope_y:.4f}\n")
        f.write(f"  - Intercept:        {intercept_y:.4f}\n")
        f.write(f"  - MAE (pixels):     {mae_y:.4f}\n")
        f.write(f"  - RMSE (pixels):    {rmse_y:.4f}\n")

    print(f"Report saved to: {report_path}")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    styles = {
        "implicit": {
            "c": "orange",
            "m": "o",
            "s": 30,
            "alpha": 0.4,
            "label": f"Implicit (n={n_impl})",
        },
        "explicit": {
            "c": "green",
            "m": "s",
            "s": 60,
            "alpha": 0.9,
            "label": f"Explicit (n={n_expl})",
        },
        "calibration": {
            "c": "blue",
            "m": "D",
            "s": 90,
            "alpha": 1.0,
            "label": f"Calibration (n={n_calib})",
        },
    }

    def plot_group(ax, x_col, y_col, source_type):
        subset = df[df["source"] == source_type]
        if subset.empty:
            return
        style = styles[source_type]
        ax.scatter(
            subset[x_col],
            subset[y_col],
            c=style["c"],
            marker=style["m"],
            s=style["s"],
            alpha=style["alpha"],
            label=style["label"],
            edgecolors="w",
            linewidth=0.5,
        )

    plot_group(axes[0], "pitch", "target_x", "implicit")
    plot_group(axes[0], "pitch", "target_x", "explicit")
    plot_group(axes[0], "pitch", "target_x", "calibration")

    sorted_idx = np.argsort(df["pitch"])
    axes[0].plot(
        df["pitch"].iloc[sorted_idx],
        pred_x.iloc[sorted_idx],
        "k--",
        alpha=0.6,
        linewidth=2,
    )

    axes[0].set_title(
        f"Relationship between ScreenX & Pitch (r={r_x:.2f})",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    axes[0].set_xlabel("Pitch (radians)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Screen X (pixels)", fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper left", frameon=True, framealpha=0.9, fancybox=True)

    plot_group(axes[1], "yaw", "target_y", "implicit")
    plot_group(axes[1], "yaw", "target_y", "explicit")
    plot_group(axes[1], "yaw", "target_y", "calibration")

    sorted_idx_y = np.argsort(df["yaw"])
    axes[1].plot(
        df["yaw"].iloc[sorted_idx_y],
        pred_y.iloc[sorted_idx_y],
        "k--",
        alpha=0.6,
        linewidth=2,
    )

    axes[1].set_title(
        f"Relationship between ScreenY & Yaw (r={r_y:.2f})",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    axes[1].set_xlabel("Yaw (radians)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Screen Y (pixels)", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper right", frameon=True, framealpha=0.9, fancybox=True)

    plt.tight_layout()
    plot_path = output_dir / "linearity_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
