"""
Benchmark script for measuring FPS/latency of trained gaze estimation models.

This script loads models from a directory and measures their inference performance
using a sample image with batch size 1.

Usage:
    uv run python benchmark.py --models_dir /path/to/models
    uv run python benchmark.py --models_dir /path/to/models --image sample.jpg
    uv run python benchmark.py --models_dir /path/to/models --device cpu
    uv run python benchmark.py --models_dir /path/to/models --num_warmup 50 --num_iterations 500
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from src.models import build_model

# Default model configuration for gaze estimation
# These are the minimal settings needed to build the model architecture
DEFAULT_NUM_BINS = 90
DEFAULT_IMAGE_SIZE = 224


def infer_backbone_from_filename(filename: str) -> str:
    """
    Infers the backbone name from a model filename.

    Expects filenames like:
    - resnet18.pth -> resnet18
    - mobilenetv3_small_100.pth -> mobilenetv3_small_100
    - efficientnet_b0_fused.pth -> efficientnet_b0
    """
    name = Path(filename).stem

    # Remove common suffixes
    for suffix in ("_best", "_latest", "_fused"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    return name


def find_models_in_directory(models_dir: str) -> list[tuple[str, str]]:
    """
    Finds all .pth model files in a directory and returns tuples of (model_path, model_name).

    Args:
        models_dir: Path to directory containing model files.

    Returns:
        List of tuples (model_path, inferred_model_name)
    """
    models = []
    models_path = Path(models_dir)

    if not models_path.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    # Find all .pth files in the directory (flat, not recursive)
    for pth_file in models_path.glob("*.pth"):
        model_path = str(pth_file)
        model_name = infer_backbone_from_filename(pth_file.name)
        models.append((model_path, model_name))

    return sorted(models, key=lambda x: x[1])


def load_and_preprocess_image(
    image_path: str | None, image_size: int = DEFAULT_IMAGE_SIZE
) -> torch.Tensor:
    """
    Loads and preprocesses an image for inference.

    If no image path is provided, creates a random tensor.

    Args:
        image_path: Path to the image file, or None for random input.
        image_size: Target image size.

    Returns:
        Preprocessed image tensor of shape (1, 3, H, W).
    """
    if image_path and Path(image_path).exists():
        image = Image.open(image_path).convert("RGB")
        image = image.resize((image_size, image_size))
        tensor = transforms.ToTensor()(image)
        tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )(tensor)
    else:
        # Create a random image if no path provided
        print("No image provided, using random input tensor.")
        tensor = torch.randn(3, image_size, image_size)

    return tensor.unsqueeze(0)  # Add batch dimension


def benchmark_model(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    num_warmup: int = 100,
    num_iterations: int = 1000,
) -> dict:
    """
    Benchmarks a model's inference performance.

    Args:
        model: The PyTorch model to benchmark.
        input_tensor: Input tensor to use for inference.
        device: Device to run on (cuda or cpu).
        num_warmup: Number of warmup iterations.
        num_iterations: Number of timed iterations.

    Returns:
        Dictionary with benchmark results.
    """
    model.eval()
    input_tensor = input_tensor.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)

    # Synchronize before timing (for CUDA)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed iterations
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            _ = model(input_tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

    latencies = np.array(latencies)

    return {
        "mean_latency_ms": float(np.mean(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "fps": float(1000.0 / np.mean(latencies)),
    }


def count_parameters(model: torch.nn.Module) -> int:
    """Counts the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def load_model_for_benchmark(
    model_path: str,
    backbone_name: str,
    device: torch.device,
) -> torch.nn.Module:
    """
    Loads a model for benchmarking.

    Args:
        model_path: Path to the .pth weights file.
        backbone_name: Name of the backbone architecture.
        device: Device to load the model on.

    Returns:
        Loaded PyTorch model.
    """
    # Minimal config needed to build the model
    config = {
        "backbone": backbone_name,
        "pretrained": False,
        "num_bins": DEFAULT_NUM_BINS,
    }

    # Check if this is a fused model (for MobileOne)
    is_fused = "_fused" in Path(model_path).stem

    backbone_kwargs = {}
    if is_fused and backbone_name.startswith("mobileone"):
        backbone_kwargs["inference_mode"] = True

    # Build and load model
    model = build_model(config, **backbone_kwargs)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()

    return model


def generate_report(
    results: list[dict],
    output_path: str,
    device_info: str,
    num_warmup: int,
    num_iterations: int,
    image_path: str | None,
):
    """
    Generates a benchmark report and saves it to disk.

    Args:
        results: List of benchmark result dictionaries.
        output_path: Path to save the report.
        device_info: String describing the device used.
        num_warmup: Number of warmup iterations used.
        num_iterations: Number of timed iterations used.
        image_path: Path to the image used for benchmarking.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Sort results by FPS (descending)
    results_sorted = sorted(results, key=lambda x: x["fps"], reverse=True)

    lines = []
    lines.append("=" * 100)
    lines.append("GAZE ESTIMATION MODEL BENCHMARK REPORT")
    lines.append("=" * 100)
    lines.append(f"Generated: {timestamp}")
    lines.append(f"Device: {device_info}")
    lines.append(f"Input image: {image_path if image_path else 'random tensor'}")
    lines.append(f"Warmup iterations: {num_warmup}")
    lines.append(f"Timed iterations: {num_iterations}")
    lines.append("Batch size: 1")
    lines.append(f"Input size: {DEFAULT_IMAGE_SIZE}x{DEFAULT_IMAGE_SIZE}")
    lines.append(f"Total models benchmarked: {len(results)}")
    lines.append("")
    lines.append("=" * 100)
    lines.append("RESULTS (sorted by FPS, descending)")
    lines.append("=" * 100)
    lines.append("")

    # Table header
    header = f"{'Rank':<6}{'Model Name':<40}{'FPS':>10}{'Mean (ms)':>12}{'Std (ms)':>10}{'P95 (ms)':>10}{'P99 (ms)':>10}{'Params':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    for i, result in enumerate(results_sorted, 1):
        if result.get("error"):
            params_str = "ERROR"
            line = (
                f"{i:<6}"
                f"{result['model_name']:<40}"
                f"{'N/A':>10}"
                f"{'N/A':>12}"
                f"{'N/A':>10}"
                f"{'N/A':>10}"
                f"{'N/A':>10}"
                f"{params_str:>12}"
            )
        else:
            params_str = f"{result['params'] / 1e6:.2f}M" if result["params"] else "N/A"
            line = (
                f"{i:<6}"
                f"{result['model_name']:<40}"
                f"{result['fps']:>10.2f}"
                f"{result['mean_latency_ms']:>12.3f}"
                f"{result['std_latency_ms']:>10.3f}"
                f"{result['p95_latency_ms']:>10.3f}"
                f"{result['p99_latency_ms']:>10.3f}"
                f"{params_str:>12}"
            )
        lines.append(line)

    lines.append("")
    lines.append("=" * 100)
    lines.append("DETAILED RESULTS")
    lines.append("=" * 100)

    for result in results_sorted:
        lines.append("")
        lines.append(f"Model: {result['model_name']}")
        lines.append(f"  Path: {result['model_path']}")

        if result.get("error"):
            lines.append(f"  Error: {result['error']}")
        else:
            lines.append(
                f"  Parameters: {result['params']:,}"
                if result["params"]
                else "  Parameters: N/A"
            )
            lines.append(f"  FPS: {result['fps']:.2f}")
            lines.append("  Latency (ms):")
            lines.append(f"    Mean: {result['mean_latency_ms']:.3f}")
            lines.append(f"    Std:  {result['std_latency_ms']:.3f}")
            lines.append(f"    Min:  {result['min_latency_ms']:.3f}")
            lines.append(f"    Max:  {result['max_latency_ms']:.3f}")
            lines.append(f"    P50:  {result['p50_latency_ms']:.3f}")
            lines.append(f"    P95:  {result['p95_latency_ms']:.3f}")
            lines.append(f"    P99:  {result['p99_latency_ms']:.3f}")

    lines.append("")
    lines.append("=" * 100)
    lines.append("END OF REPORT")
    lines.append("=" * 100)

    report_content = "\n".join(lines)

    # Save to file
    with open(output_path, "w") as f:
        f.write(report_content)

    # Also print to console
    print(report_content)
    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FPS/latency of trained gaze estimation models."
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="Path to directory containing model .pth files.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a sample image for inference. If not provided, uses random input.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the benchmark report. Defaults to benchmark_report_<timestamp>.txt",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=100,
        help="Number of warmup iterations (default: 100).",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=1000,
        help="Number of timed iterations (default: 1000).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda or cpu). Defaults to cuda if available.",
    )
    args = parser.parse_args()

    # Set up device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_info = str(device)
    if device.type == "cuda":
        device_info += f" ({torch.cuda.get_device_name(0)})"

    print(f"Using device: {device_info}")

    # Load and preprocess input image
    input_tensor = load_and_preprocess_image(args.image, DEFAULT_IMAGE_SIZE)
    print(f"Input tensor shape: {input_tensor.shape}")

    # Find models
    print(f"\nSearching for models in: {args.models_dir}")
    models = find_models_in_directory(args.models_dir)

    if not models:
        print("No model files found!")
        return

    print(f"Found {len(models)} model(s) to benchmark:\n")
    for model_path, model_name in models:
        print(f"  - {model_name}")
    print()

    # Benchmark each model
    results = []
    for i, (model_path, model_name) in enumerate(models, 1):
        print(f"[{i}/{len(models)}] Benchmarking: {model_name}")

        result = {
            "model_name": model_name,
            "model_path": model_path,
            "params": None,
            "error": None,
        }

        try:
            # Load model
            model = load_model_for_benchmark(model_path, model_name, device)
            result["params"] = count_parameters(model)

            # Run benchmark
            benchmark_results = benchmark_model(
                model,
                input_tensor,
                device,
                num_warmup=args.num_warmup,
                num_iterations=args.num_iterations,
            )
            result.update(benchmark_results)

            print(
                f"  FPS: {result['fps']:.2f}, Mean latency: {result['mean_latency_ms']:.3f}ms"
            )

            # Clean up
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Error: {e}")
            result["error"] = str(e)
            result["fps"] = 0
            result["mean_latency_ms"] = float("inf")
            result["std_latency_ms"] = 0
            result["min_latency_ms"] = float("inf")
            result["max_latency_ms"] = float("inf")
            result["p50_latency_ms"] = float("inf")
            result["p95_latency_ms"] = float("inf")
            result["p99_latency_ms"] = float("inf")

        results.append(result)

    # Generate output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = f"benchmark_report_{timestamp}.txt"

    # Generate report
    print("\n" + "=" * 50)
    print("Generating report...")
    generate_report(
        results,
        output_path,
        device_info,
        args.num_warmup,
        args.num_iterations,
        args.image,
    )


if __name__ == "__main__":
    main()
