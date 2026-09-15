"""
====================================================
RetinaSense Logger
====================================================

"""

import logging
from pathlib import Path
from core.paths import LOGS

# Create logs directory if it doesn't exist
LOGS.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOGS / "retinasense.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("RetinaSense")