import sys
from pathlib import Path

# Ensure the project root is importable so tests can `import src.<module>`
# regardless of the directory pytest is invoked from.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
