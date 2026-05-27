import os


class Config:
    """Feature flags controlled via environment variables."""

    def __init__(self):
        master = os.environ.get("OMNIXPU_ENABLE", "1") != "0"
        self.attention = master and os.environ.get("OMNIXPU_ATTENTION", "1") != "0"
        self.rope = master and os.environ.get("OMNIXPU_ROPE", "1") != "0"
        self.norm = master and os.environ.get("OMNIXPU_NORM", "1") != "0"
        self.fp8_gemm = master and os.environ.get("OMNIXPU_FP8_GEMM", "1") != "0"
        self.fp8_neg_zero_fix = master and os.environ.get("OMNIXPU_FP8_NEG_ZERO_FIX", "1") != "0"
        self.interpolate_fix = master and os.environ.get("OMNIXPU_INTERPOLATE_FIX", "1") != "0"


config = Config()
