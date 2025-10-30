import os
import sys

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# This ensures that the 'inference' module can be found by Python
# when running this script from the project root directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_collector.app import GazeDataCollectionApp

if __name__ == "__main__":
    app_instance = GazeDataCollectionApp()
    sys.exit(app_instance.run())
