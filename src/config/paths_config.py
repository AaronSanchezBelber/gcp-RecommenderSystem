import os

# ===================== ROOT DEL PROYECTO =====================

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

# ===================== DIRECTORIOS BASE =====================

ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
RAW_DIR = os.path.join(ARTIFACTS_DIR, "raw")
PROCESSED_DIR = os.path.join(ARTIFACTS_DIR, "processed")
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "model")
WEIGHTS_DIR = os.path.join(ARTIFACTS_DIR, "weights")
CHECKPOINT_DIR = os.path.join(ARTIFACTS_DIR, "model_checkpoint")

# ===================== CONFIG =====================

CONFIG_PATH = os.path.join(
    ROOT_DIR,
    "src",
    "config",
    "config.yaml"
)

# ===================== RAW DATA =====================

ANIMELIST_CSV = os.path.join(RAW_DIR, "animelist.csv")
ANIME_CSV = os.path.join(RAW_DIR, "anime.csv")
ANIMESYNOPSIS_CSV = os.path.join(RAW_DIR, "anime_with_synopsis.csv")

# ===================== PROCESSED DATA =====================

X_TRAIN_ARRAY = os.path.join(PROCESSED_DIR, "X_train_array.pkl")
X_TEST_ARRAY = os.path.join(PROCESSED_DIR, "X_test_array.pkl")
Y_TRAIN = os.path.join(PROCESSED_DIR, "y_train.pkl")
Y_TEST = os.path.join(PROCESSED_DIR, "y_test.pkl")

RATING_DF = os.path.join(PROCESSED_DIR, "rating_df.csv")
DF = os.path.join(PROCESSED_DIR, "anime_df.csv")
SYNOPSIS_DF = os.path.join(PROCESSED_DIR, "synopsis_df.csv")

USER2USER_ENCODED = os.path.join(PROCESSED_DIR, "user2user_encoded.pkl")
USER2USER_DECODED = os.path.join(PROCESSED_DIR, "user2user_decoded.pkl")

ANIME2ANIME_ENCODED = os.path.join(PROCESSED_DIR, "anim2anime_encoded.pkl")
ANIME2ANIME_DECODED = os.path.join(PROCESSED_DIR, "anim2anime_decoded.pkl")

# ===================== MODEL =====================

MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")

ANIME_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "anime_weights.pkl")
USER_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "user_weights.pkl")

CHECKPOINT_FILE_PATH = os.path.join(
    CHECKPOINT_DIR,
    "weights.weights.h5"
)
