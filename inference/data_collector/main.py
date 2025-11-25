import sys
from pathlib import Path

from dotenv import load_dotenv

# This ensures that the 'inference' module can be found by Python
if getattr(sys, "frozen", False):
    # Running as PyInstaller bundle
    base_path = Path(getattr(sys, "_MEIPASS", "."))
else:
    # Running as script
    base_path = Path(__file__).parent.parent

sys.path.insert(0, str(base_path))

from data_collector.app import GazeDataCollectionApp

# Load environment variables from .env file
env_file = base_path / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
    print(f"Loaded .env from {env_file}")
else:
    print(f"Warning: .env not found at {env_file}. Using environment variables.")
    load_dotenv()  # Fall back to current directory

if __name__ == "__main__":
    app_instance = GazeDataCollectionApp()
    sys.exit(app_instance.run())
