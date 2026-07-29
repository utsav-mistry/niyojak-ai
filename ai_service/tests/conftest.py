"""
conftest.py — pytest path bootstrap for the ai_service test suite.

Adds ai_service/app and ai_service/train to sys.path so that the test
module can import feature_store, model, main, and train_model directly
without installing them as packages.
"""
import sys
import os

# Resolve ai_service root regardless of where pytest is invoked from
_ai_service_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for _subdir in ("app", "train"):
    _path = os.path.join(_ai_service_root, _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)
