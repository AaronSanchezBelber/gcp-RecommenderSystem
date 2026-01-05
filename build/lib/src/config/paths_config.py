import os

# Raíz del proyecto
ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

# Paths absolutos
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
RAW_DIR = os.path.join(ARTIFACTS_DIR, "raw")

CONFIG_PATH = os.path.join(
    ROOT_DIR,
    "src",
    "config",
    "config.yaml"
)
