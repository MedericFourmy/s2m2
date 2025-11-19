import os
from pathlib import Path

S2M2_PRETRAINED_WEIGHTS_PATH = Path(os.getenv("S2M2_PRETRAINED_WEIGHTS_PATH", "pretrain_weights"))
