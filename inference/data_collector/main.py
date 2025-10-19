import os
import sys

# This ensures that the 'inference' module can be found by Python
# when running this script from the project root directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_collector.app import SystemTrayApp

if __name__ == "__main__":
    app_instance = SystemTrayApp()
    sys.exit(app_instance.run())
