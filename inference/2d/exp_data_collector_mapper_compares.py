import argparse
from pathlib import Path

import exp_data_collector as collector
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

from src.inference import GazePipeline2D, GazePipeline3D


def extract_datasets(webcam_path, metadata, gaze_pipeline_3d, context_frames=5):
    """
    Uses the existing pipeline to extract:
    1. Training Data (from Calibration 9-points)
    2. Test Data (from Explicit Clicks)
    """
    print("Step 1: Extracting Training Data (Calibration)...")
    base_mapper = collector.train_and_get_initial_mapper(
        webcam_path, metadata, gaze_pipeline_3d, context_frames=context_frames
    )

    X_train = np.array(base_mapper.initial_feature_vectors)
    y_train = np.array(base_mapper.initial_screen_points)

    print(f"  -> Training samples: {len(X_train)}")

    print("Step 2: Extracting Test Data (Explicit Clicks)...")
    explicit_clicks = [c for c in metadata["clicks"] if c["type"] == "explicit"]

    cap = collector.cv2.VideoCapture(str(webcam_path))
    fps = cap.get(collector.cv2.CAP_PROP_FPS)

    pipeline_2d = GazePipeline2D(gaze_pipeline_3d, base_mapper, ["pitch", "yaw"])

    X_test = []
    y_test = []

    frame_to_clicks = {}
    for click in explicit_clicks:
        click_frame = int(click["videoTimestamp"] * fps / 1000)
        window = range(
            max(0, click_frame - context_frames * 2), click_frame + context_frames
        )
        for f in window:
            if f not in frame_to_clicks:
                frame_to_clicks[f] = []
            frame_to_clicks[f].append(click)

    total_frames = int(cap.get(collector.cv2.CAP_PROP_FRAME_COUNT))

    for frame_idx in tqdm(range(total_frames), desc="Extracting Test Data"):
        if frame_idx not in frame_to_clicks:
            continue

        cap.set(collector.cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        results_3d = gaze_pipeline_3d(frame)
        if results_3d:
            features = pipeline_2d.extract_feature_vector(results_3d[0])
            for click in frame_to_clicks[frame_idx]:
                X_test.append(features)
                y_test.append([click["screenX"], click["screenY"]])

    cap.release()
    return X_train, y_train, np.array(X_test), np.array(y_test)


def train_and_evaluate_models(X_train, y_train, X_test, y_test, output_dir):
    """
    Trains comprehensive suite of regressors and evaluates them.
    """

    # --- MODEL DEFINITIONS ---
    # use MultiOutputRegressor for models that don't support 2D target (x,y) natively
    models = {
        # 1. Linear Family
        "Linear Regression (Ours)": LinearRegression(),
        # "Ridge (alpha=1.0)": Ridge(alpha=1.0),
        # 2. Polynomial
        "Polynomial (Deg 2)": make_pipeline(PolynomialFeatures(2), LinearRegression()),
        # 3. Neighbors
        "KNN (k=3)": KNeighborsRegressor(n_neighbors=3),
        # 4. SVM
        # C=10 is moderate regularization. Too high = overfit, Too low = underfit.
        "SVR (RBF, C=10)": MultiOutputRegressor(
            make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.1))
        ),
        # 5. Trees (Constrained)
        # max_depth=5 prevents the tree from memorizing every single point perfectly
        "Decision Tree (Depth=5)": MultiOutputRegressor(
            DecisionTreeRegressor(max_depth=5, random_state=42)
        ),
        # n_estimators=100 is standard.
        "Random Forest (n=100)": MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, random_state=42)
        ),
        # 6. Neural Network (Tiny)
        # (16, 16) is very small, which is CORRECT for small data.
        # Large networks (100, 100) would be absurd here.
        "MLP (2-Layer (16, 16))": MultiOutputRegressor(
            make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(16, 16), max_iter=5000, random_state=42
                ),
            )
        ),
    }

    results = []

    print("\nStep 3: Benchmarking Models...")
    for name, model in models.items():
        print(f"  Training {name}...")
        try:
            # Train
            model.fit(X_train, y_train)

            # Predict
            preds = model.predict(X_test)

            # Calculate Error (Euclidean Distance)
            errors = np.sqrt(np.sum((preds - y_test) ** 2, axis=1))
            mae = np.mean(errors)

            print(f"    -> MAE: {mae:.2f} px")

            # Store for plotting
            for err in errors:
                results.append({"Model": name, "Error (px)": err})
        except Exception as e:
            print(f"    -> Failed: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save Raw Data
    df.to_csv(output_dir / "mapper_comparison_raw.csv", index=False)

    # --- Visualizations ---
    sns.set_theme(style="whitegrid")

    # 1. Bar Plot (Mean Error)
    plt.figure(figsize=(12, 4))
    # Order by mean error (ascending)
    order = df.groupby("Model")["Error (px)"].mean().sort_values().index

    sns.barplot(
        data=df,
        x="Error (px)",
        y="Model",
        order=order,
        errorbar="sd",
        palette="viridis",
    )
    plt.title("Mean Gaze Error by Regression Models")
    plt.xlabel("Mean Euclidean Error (pixels)")
    plt.tight_layout()
    plt.savefig(output_dir / "mapper_comparison_bar.png", dpi=300)
    plt.close()

    # 2. Box Plot (Distribution/Outliers)
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=df,
        x="Error (px)",
        y="Model",
        order=order,
        palette="viridis",
        showfliers=False,
    )
    plt.title("Error Distribution by Model (Lower is Better)")
    plt.xlabel("Euclidean Error (pixels)")
    plt.tight_layout()
    plt.savefig(output_dir / "mapper_comparison_box.png", dpi=300)
    plt.close()

    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="mapper_comparison")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load Metadata & Model
    metadata = collector.load_metadata(data_dir / "metadata.json")
    pipeline_3d = GazePipeline3D(
        weights_path=args.weights, device=args.device, smooth_facebbox=True
    )

    # Run
    X_train, y_train, X_test, y_test = extract_datasets(
        data_dir / "webcam.mp4", metadata, pipeline_3d
    )

    if len(X_test) == 0:
        print("Error: No test data found (no explicit clicks in metadata?)")
        return

    train_and_evaluate_models(X_train, y_train, X_test, y_test, output_dir)


if __name__ == "__main__":
    main()
