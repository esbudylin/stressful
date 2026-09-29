from importlib.resources import files

DATA_DIRECTORY = files("stressful") / "data"

ACCENT_DICT_DIR = DATA_DIRECTORY / "dicts"
ACCENT_DICT_PATHS = tuple(
    ACCENT_DICT_DIR / name for name in ("accent.dic", "accent1.dic", "accent2.dic")
)

SILERO_MODEL_DIR = DATA_DIRECTORY / "models"

STRESS_TOKEN = "+"
