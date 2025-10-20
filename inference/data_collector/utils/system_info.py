import platform

import psutil
import torch


def bytes_to_gb(b):
    return round(b / (1024**3), 2)


def get_system_info() -> dict:
    """
    Gathers detailed OS, CPU, and RAM information
    """
    print("Gathering detailed system information...")

    # --- CPU Information ---
    cpu_info_data = {}
    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()

        # Get a meaningful initial CPU usage value
        psutil.cpu_percent(interval=0.5)

        cpu_info_data = {
            "brand": info.get("brand_raw", "N/A"),
            "arch": info.get("arch_string_raw", "N/A"),
            "base_speed_ghz": round(info.get("hz_advertised", (0, 0))[0] / 1e9, 2),
            "flags": info.get("flags", []),
            "l2_cache_size_kb": info.get("l2_cache_size", 0) // 1024,
            "l3_cache_size_kb": info.get("l3_cache_size", 0) // 1024,
            "physical_cores": psutil.cpu_count(logical=False),
            "total_cores": psutil.cpu_count(logical=True),
            "current_usage_percent": psutil.cpu_percent(),
        }
    except Exception as e:
        print(
            f"Warning: Could not get detailed CPU info via cpuinfo. Falling back. Error: {e}"
        )
        cpu_info_data = {
            "brand": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "total_cores": psutil.cpu_count(logical=True),
            "error": str(e),
        }

    # --- RAM Information ---
    ram_info_data = {}
    try:
        mem = psutil.virtual_memory()
        ram_info_data = {
            "total_gb": bytes_to_gb(mem.total),
            "available_gb": bytes_to_gb(mem.available),
            "usage_percent": mem.percent,
        }
    except Exception as e:
        ram_info_data = {"error": str(e)}

    # --- Aggregate all information ---
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cpu": cpu_info_data,
        "ram": ram_info_data,
    }
