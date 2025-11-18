# Global config 
DATA_DIR = "data"
ART_DIR  = "artifacts"
OUT_DIR  = "outputs"

# Symbols to exclude
EXCLUDE_SYMBOLS = {"USDT", "USDC"}

# IO / Write mode
SPARK_WRITE_MODE = "overwrite"

# === Feature/z-score config (no CLI args) ===
USE_LOG_RET      = True     # log returns for stability
WINSOR_LO        = 0.025    # per-coin lower quantile for winsorization
WINSOR_HI        = 0.975    # per-coin upper quantile for winsorization
SIGMA_FLOOR      = 1e-4     # min std-dev to allow z
MIN_PRIOR_OBS_7  = 5        # must be <= 7
MIN_PRIOR_OBS_30 = 10
BURN_IN_DAYS     = 14
TOPK             = 10
