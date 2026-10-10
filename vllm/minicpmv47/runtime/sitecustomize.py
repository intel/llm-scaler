# SPDX-License-Identifier: Apache-2.0
import os
import sys
import traceback

if os.environ.get("MINICPMV47_RUNTIME") == "1":
    try:
        from minicpmv47_runtime import install

        install()
    except Exception:  # noqa: BLE001 - every overlay failure must abort startup
        # Python otherwise ignores a sitecustomize failure and serves stock ops.
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        os._exit(1)
