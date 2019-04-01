import os
import sys

def get(relative_path):
    """Get a file resource, even when bundled into a single executable."""
    try:
        base_path = sys._MEIPASS
    except:
        base_path = os.path.abspath('.')
    return os.path.join(base_path, relative_path)
