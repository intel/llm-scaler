# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Regression checks for the build gate's narrow metadata exception policy."""
import unittest

from check_runtime_dependencies import allowed_issue


class DependencyPolicyTests(unittest.TestCase):
    def test_known_missing_backend_is_exempt(self):
        self.assertTrue(allowed_issue("lmcache 0.4.7 requires cufile-python, which is not installed."))
        self.assertTrue(allowed_issue("torch 2.12.0+xpu requires oneccl, which is not installed."))

    def test_actual_sdk_conflict_is_rejected(self):
        self.assertFalse(allowed_issue(
            "opentelemetry-exporter-otlp-common 0.66b1 has requirement "
            "opentelemetry-sdk~=1.45.1, but you have opentelemetry-sdk 1.40.0."
        ))

    def test_optional_backend_version_conflict_is_rejected(self):
        self.assertFalse(allowed_issue(
            "lmcache 0.4.7 has requirement redis>=5, but you have redis 4.0.0."
        ))

    def test_new_missing_dependency_is_rejected(self):
        self.assertFalse(allowed_issue("lmcache 0.4.7 requires huggingface-hub, which is not installed."))

    def test_new_owner_or_version_requires_review(self):
        self.assertFalse(allowed_issue("vllm 0.26.0 requires redis, which is not installed."))
        self.assertFalse(allowed_issue("lmcache 0.5.0 requires redis, which is not installed."))
        self.assertFalse(allowed_issue("torch 2.13.0+xpu requires oneccl, which is not installed."))

    def test_decord_exception_is_exact(self):
        self.assertTrue(allowed_issue("decord 0.6.0 is not supported on this platform"))
        self.assertFalse(allowed_issue("decord 0.7.0 is not supported on this platform"))


if __name__ == "__main__":
    unittest.main()
