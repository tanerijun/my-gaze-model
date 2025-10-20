import platform

import psutil


def get_system_info() -> dict:
    """Gathers detailed hardware and OS information."""

    # Helper to convert bytes to a readable format (GB)
    def bytes_to_gb(b):
        return round(b / (1024**3), 2)

    try:
        # Get initial CPU usage over a short interval to get a meaningful value
        psutil.cpu_percent(interval=1)

        mem = psutil.virtual_memory()

        info = {
            "os": f"{platform.system()} {platform.release()}",
            "cpu": {
                "name": platform.processor(),
                "physical_cores": psutil.cpu_count(logical=False),
                "total_cores": psutil.cpu_count(logical=True),
                "current_usage_percent": psutil.cpu_percent(),
            },
            "ram": {
                "total_gb": bytes_to_gb(mem.total),
                "available_gb": bytes_to_gb(mem.available),
                "usage_percent": mem.percent,
            },
        }
        return info
    except Exception as e:
        print(f"Could not gather all system info: {e}")
        return {"error": str(e)}
