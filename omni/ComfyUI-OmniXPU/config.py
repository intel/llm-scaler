import os


class Config:
    """Feature flags controlled via environment variables.

    Generic XPU operators are selected by comfy_kitchen and therefore do not
    have custom-node flags here.  Legacy global workarounds are opt-in.
    """

    def __init__(self):
        master = os.environ.get("OMNIXPU_ENABLE", "1") != "0"
        self.attention = master and os.environ.get("OMNIXPU_ATTENTION", "1") != "0"
        self.norm = master and os.environ.get("OMNIXPU_NORM", "1") != "0"
        self.fp8_gemm = master and os.environ.get("OMNIXPU_FP8_GEMM", "1") != "0"
        self.int8_ffn = master and os.environ.get("OMNIXPU_INT8_FFN", "1") != "0"
        self.interpolate_fix = (
            master and os.environ.get("OMNIXPU_INTERPOLATE_FIX", "0") != "0"
        )
        self.median_fix = (
            master and os.environ.get("OMNIXPU_MEDIAN_FIX", "0") != "0"
        )


config = Config()
