"""
Benchmark script for measuring FPS/latency and memory usage of trained gaze estimation models.

This script loads models from a directory and measures their inference performance
and memory consumption using a sample image with batch size 1.

Usage:
    uv run python latency_benchmark.py --models_dir /path/to/models --image sample.jpg
    uv run python latency_benchmark.py --models_dir /path/to/models --image sample.jpg --num_warmup 50 --num_iterations 1000
"""

import argparse
import gc
import os
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch
import torchvision.transforms as transforms
from PIL import Image

from src.models import build_model

# Default model configuration for gaze estimation
# These are the minimal settings needed to build the model architecture
DEFAULT_NUM_BINS = 90
DEFAULT_IMAGE_SIZE = 224


def get_process_memory_mb() -> float:
    """Returns the current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Calculates the model size in MB based on parameter and buffer storage.

    Args:
        model: PyTorch model.

    Returns:
        Model size in MB.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size_bytes = param_size + buffer_size
    return total_size_bytes / (1024 * 1024)


def infer_backbone_from_filename(filename: str) -> str:
    """
    Infers the backbone name from a model filename.

    Expects filenames like:
    - resnet18.pth -> resnet18
    - mobilenetv3_small_100.pth -> mobilenetv3_small_100
    - efficientnet_b0_fused.pth -> efficientnet_b0
    """
    name = Path(filename).stem

    # Remove the 'gaze360_' prefix if it exists
    if name.startswith("gaze360_"):
        name = name[len("gaze360_") :]

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
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
        image = image.resize((image_size, image_size))
        tensor = transforms.ToTensor()(image)
        tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )(tensor)
    else:
        if image_path:
            print(f"Image file not found: {image_path}")
        print("No image provided or file not found, using random input tensor.")
        tensor = torch.randn(3, image_size, image_size)

    return tensor.unsqueeze(0)  # Add batch dimension


def benchmark_model(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = 100,
    num_iterations: int = 1000,
) -> dict:
    """
    Benchmarks a model's inference performance and memory usage on CPU.

    Args:
        model: The PyTorch model to benchmark.
        input_tensor: Input tensor to use for inference.
        num_warmup: Number of warmup iterations.
        num_iterations: Number of timed iterations.

    Returns:
        Dictionary with benchmark results including latency and memory metrics.
    """
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)

    # Force garbage collection before memory measurement
    gc.collect()

    # Get baseline memory
    baseline_memory_mb = get_process_memory_mb()

    # Start memory tracking with tracemalloc
    tracemalloc.start()

    # Timed iterations with memory tracking
    latencies = []
    peak_memory_during_inference = baseline_memory_mb

    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            _ = model(input_tensor)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

            # Track peak memory
            current_memory = get_process_memory_mb()
            peak_memory_during_inference = max(
                peak_memory_during_inference, current_memory
            )

    # Get tracemalloc stats
    current_traced, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()

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
        # Memory metrics
        "baseline_memory_mb": baseline_memory_mb,
        "peak_memory_mb": peak_memory_during_inference,
        "memory_increase_mb": peak_memory_during_inference - baseline_memory_mb,
        "peak_traced_memory_mb": peak_traced / (1024 * 1024),
    }


def count_parameters(model: torch.nn.Module) -> int:
    """Counts the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model_for_benchmark(
    model_path: str,
    backbone_name: str,
) -> torch.nn.Module:
    """
    Loads a model for benchmarking on CPU.

    Args:
        model_path: Path to the .pth weights file.
        backbone_name: Name of the backbone architecture.

    Returns:
        Loaded PyTorch model on CPU.
    """
    device = torch.device("cpu")

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


def get_file_size_mb(file_path: str) -> float:
    """Returns the file size in MB."""
    return os.path.getsize(file_path) / (1024 * 1024)


def generate_report(
    results: list[dict],
    output_path: str,
    system_info: dict,
    num_warmup: int,
    num_iterations: int,
    image_path: str | None,
):
    """
    Generates a benchmark report and saves it to disk.

    Args:
        results: List of benchmark result dictionaries.
        output_path: Path to save the report.
        system_info: Dictionary with system information.
        num_warmup: Number of warmup iterations used.
        num_iterations: Number of timed iterations used.
        image_path: Path to the image used for benchmarking.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Sort results by FPS (descending)
    results_sorted = sorted(results, key=lambda x: x["fps"], reverse=True)

    lines = []
    lines.append("=" * 120)
    lines.append("GAZE ESTIMATION MODEL BENCHMARK REPORT")
    lines.append("=" * 120)
    lines.append(f"Generated: {timestamp}")
    lines.append("")
    lines.append("SYSTEM INFORMATION")
    lines.append("-" * 40)
    lines.append("Device: CPU")
    lines.append(f"CPU: {system_info['cpu_info']}")
    lines.append(f"CPU Cores (Physical): {system_info['cpu_cores_physical']}")
    lines.append(f"CPU Cores (Logical): {system_info['cpu_cores_logical']}")
    lines.append(f"Total System Memory: {system_info['total_memory_gb']:.2f} GB")
    lines.append(
        f"Available Memory at Start: {system_info['available_memory_gb']:.2f} GB"
    )
    lines.append(f"PyTorch Version: {system_info['pytorch_version']}")
    lines.append("")
    lines.append("BENCHMARK CONFIGURATION")
    lines.append("-" * 40)
    lines.append(f"Input image: {image_path if image_path else 'random tensor'}")
    lines.append(f"Warmup iterations: {num_warmup}")
    lines.append(f"Timed iterations: {num_iterations}")
    lines.append("Batch size: 1")
    lines.append(f"Input size: {DEFAULT_IMAGE_SIZE}x{DEFAULT_IMAGE_SIZE}")
    lines.append(f"Total models benchmarked: {len(results)}")
    lines.append("")
    lines.append("=" * 120)
    lines.append("RESULTS SUMMARY (sorted by FPS, descending)")
    lines.append("=" * 120)
    lines.append("")

    # Table header
    header = (
        f"{'Rank':<6}"
        f"{'Model Name':<35}"
        f"{'FPS':>10}"
        f"{'Mean (ms)':>12}"
        f"{'P95 (ms)':>10}"
        f"{'Params':>12}"
        f"{'Model (MB)':>12}"
        f"{'File (MB)':>12}"
        f"{'Peak Mem (MB)':>14}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for i, result in enumerate(results_sorted, 1):
        if result.get("error"):
            line = (
                f"{i:<6}"
                f"{result['model_name']:<35}"
                f"{'ERROR':>10}"
                f"{'N/A':>12}"
                f"{'N/A':>10}"
                f"{'N/A':>12}"
                f"{'N/A':>12}"
                f"{'N/A':>12}"
                f"{'N/A':>14}"
            )
        else:
            params_str = f"{result['params'] / 1e6:.2f}M" if result["params"] else "N/A"
            model_size_str = (
                f"{result['model_size_mb']:.2f}"
                if result.get("model_size_mb")
                else "N/A"
            )
            file_size_str = (
                f"{result['file_size_mb']:.2f}" if result.get("file_size_mb") else "N/A"
            )
            peak_mem_str = (
                f"{result['peak_memory_mb']:.1f}"
                if result.get("peak_memory_mb")
                else "N/A"
            )
            line = (
                f"{i:<6}"
                f"{result['model_name']:<35}"
                f"{result['fps']:>10.2f}"
                f"{result['mean_latency_ms']:>12.3f}"
                f"{result['p95_latency_ms']:>10.3f}"
                f"{params_str:>12}"
                f"{model_size_str:>12}"
                f"{file_size_str:>12}"
                f"{peak_mem_str:>14}"
            )
        lines.append(line)

    lines.append("")
    lines.append("=" * 120)
    lines.append("DETAILED RESULTS")
    lines.append("=" * 120)

    for result in results_sorted:
        lines.append("")
        lines.append(f"Model: {result['model_name']}")
        lines.append(f"  Path: {result['model_path']}")

        if result.get("error"):
            lines.append(f"  Error: {result['error']}")
        else:
            lines.append("")
            lines.append("  Model Information:")
            lines.append(
                f"    Total Parameters: {result['params']:,}"
                if result["params"] is not None
                else "    Total Parameters: N/A"
            )
            lines.append(
                f"    Trainable Parameters: {result['trainable_params']:,}"
                if result.get("trainable_params") is not None
                else "    Trainable Parameters: N/A"
            )
            lines.append(
                f"    Model Size (in memory): {result['model_size_mb']:.2f} MB"
                if result.get("model_size_mb") is not None
                else "    Model Size (in memory): N/A"
            )
            lines.append(
                f"    File Size (on disk): {result['file_size_mb']:.2f} MB"
                if result.get("file_size_mb") is not None
                else "    File Size (on disk): N/A"
            )

            lines.append("")
            lines.append("  Performance Metrics:")
            lines.append(f"    FPS: {result['fps']:.2f}")
            lines.append("    Latency (ms):")
            lines.append(f"      Mean: {result['mean_latency_ms']:.3f}")
            lines.append(f"      Std:  {result['std_latency_ms']:.3f}")
            lines.append(f"      Min:  {result['min_latency_ms']:.3f}")
            lines.append(f"      Max:  {result['max_latency_ms']:.3f}")
            lines.append(f"      P50:  {result['p50_latency_ms']:.3f}")
            lines.append(f"      P95:  {result['p95_latency_ms']:.3f}")
            lines.append(f"      P99:  {result['p99_latency_ms']:.3f}")

            lines.append("")
            lines.append("  Memory Metrics:")
            lines.append(
                f"    Baseline Process Memory: {result['baseline_memory_mb']:.1f} MB"
                if result.get("baseline_memory_mb") is not None
                else "    Baseline Process Memory: N/A"
            )
            lines.append(
                f"    Peak Process Memory: {result['peak_memory_mb']:.1f} MB"
                if result.get("peak_memory_mb") is not None
                else "    Peak Process Memory: N/A"
            )
            lines.append(
                f"    Memory Increase During Inference: {result['memory_increase_mb']:.1f} MB"
                if result.get("memory_increase_mb") is not None
                else "    Memory Increase During Inference: N/A"
            )
            lines.append(
                f"    Peak Traced Memory (tracemalloc): {result['peak_traced_memory_mb']:.2f} MB"
                if result.get("peak_traced_memory_mb") is not None
                else "    Peak Traced Memory (tracemalloc): N/A"
            )

    lines.append("")
    lines.append("=" * 120)
    lines.append("END OF REPORT")
    lines.append("=" * 120)

    report_content = "\n".join(lines)

    # Save to file
    with open(output_path, "w") as f:
        f.write(report_content)

    # Also print to console
    print(report_content)
    print(f"\nReport saved to: {output_path}")


def get_system_info() -> dict:
    """Gathers system information for the benchmark report."""
    import platform

    memory = psutil.virtual_memory()

    # Try to get CPU info
    cpu_info = platform.processor()
    if not cpu_info:
        cpu_info = platform.machine()

    return {
        "cpu_info": cpu_info,
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "total_memory_gb": memory.total / (1024**3),
        "available_memory_gb": memory.available / (1024**3),
        "pytorch_version": torch.__version__,
        "platform": platform.platform(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FPS/latency and memory usage of trained gaze estimation models (CPU only)."
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
        required=True,
        help="Path to a sample image for inference.",
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
    args = parser.parse_args()

    # CPU only - no device argument needed
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Get system info
    system_info = get_system_info()
    print(f"CPU: {system_info['cpu_info']}")
    print(f"Total Memory: {system_info['total_memory_gb']:.2f} GB")
    print(f"PyTorch Version: {system_info['pytorch_version']}")

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
            "trainable_params": None,
            "model_size_mb": None,
            "file_size_mb": None,
            "error": None,
        }

        try:
            # Force garbage collection before loading new model
            gc.collect()

            # Get file size on disk
            result["file_size_mb"] = get_file_size_mb(model_path)

            # Load model
            model = load_model_for_benchmark(model_path, model_name)
            result["params"] = count_parameters(model)
            result["trainable_params"] = count_trainable_parameters(model)
            result["model_size_mb"] = get_model_size_mb(model)

            # Run benchmark
            benchmark_results = benchmark_model(
                model,
                input_tensor,
                num_warmup=args.num_warmup,
                num_iterations=args.num_iterations,
            )
            result.update(benchmark_results)

            print(
                f"  FPS: {result['fps']:.2f}, "
                f"Mean latency: {result['mean_latency_ms']:.3f}ms, "
                f"Peak memory: {result['peak_memory_mb']:.1f}MB"
            )

            # Clean up
            del model
            gc.collect()

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
        system_info,
        args.num_warmup,
        args.num_iterations,
        args.image,
    )


if __name__ == "__main__":
    main()
