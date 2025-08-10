import os
from pathlib import Path
import flowander

PROJECT_DIR = Path(flowander.__file__).parent.parent.parent
DATA_DIR = Path(
    os.environ.get("FLOWANDER_DATA", Path(PROJECT_DIR) / "data"),
)
assert DATA_DIR.exists(), (
    f"Create directory {DATA_DIR} or set 'FLOWANDER_DATA' env variable"
)

WEIGHTS_DIR = DATA_DIR / "weights"
FIGS_DIR = DATA_DIR / "figs"