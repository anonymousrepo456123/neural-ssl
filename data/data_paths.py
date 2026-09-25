"""Dataset path configuration.

Set each variable to the absolute path of the corresponding preprocessed
pickle file (or directory) on your system before running pretrain.py or
finetune.py. 
"""

# ── T12 / T15 speech ──────────────────────────────────────────────────────
TRAIN_CARD_DATASET = "/path/to/card_t15_train.pkl"
VAL_CARD_DATASET   = "/path/to/card_t15_val.pkl"
TEST_CARD_DATASET  = "/path/to/card_t15_test.pkl"

TRAIN_WILLET_DATASET       = "/path/to/willet_t12_train.pkl"
TEST_WILLET_DATASET        = "/path/to/willet_t12_test.pkl"
COMPETITION_WILLET_DATASET = "/path/to/willet_t12_competition.pkl"

# ── Handwriting ───────────────────────────────────────────────────────────
TRAIN_HANDWRITING_DATASET = "/path/to/handwriting_train.pkl"
TEST_HANDWRITING_DATASET  = "/path/to/handwriting_test.pkl"

TRAIN_FAN_HANDWRITING_DATSET = "/path/to/fan_handwriting_train.pkl"
VAL_FAN_HANDWRITING_DATSET   = "/path/to/fan_handwriting_val.pkl"
TEST_FAN_HANDWRITING_DATSET  = "/path/to/fan_handwriting_test.pkl"

# ── Wairagkar ─────────────────────────────────────────────────────────────
TRAIN_WAIRAGKAR_DATASET = "/path/to/wairagkar_train.pkl"

# ── Kunz (directory containing kunz_t12/t15/t16/t17 .pkl files) ──────────
KUNZ_DATASET_PATH = "/path/to/kunz_data_dir"

# ── Neural Pile -------------------------------───────────────────────────
TRAIN_NEURALPILE_DATASET = "/path/to/neuralpile_processed_train"
TEST_NEURALPILE_DATASET  = "/path/to/neuralpile_processed_test"

# ── Jude speech / typing ──────────────────────────────────────────────────
JUDE_SPEECH_LARGE_TRAIN = "/path/to/jude_speech_large_train.pkl"
JUDE_TYPING_T17         = "/path/to/jude_typing_t17.pkl"
JUDE_TYPING_T18         = "/path/to/jude_typing_t18.pkl"
