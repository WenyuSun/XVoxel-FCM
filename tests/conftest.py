# -*- coding: utf-8 -*-
"""pytest configuration for XVoxel-FCM tests.

Ensures the repo root is on ``sys.path`` so tests can use absolute imports
(``from xvoxel import ...``, ``from fcm import ...``, ``from fluid import ...``)
and registers the custom ``slow`` mark.
"""
import os
import sys

# Ensure the repo root is on sys.path so tests can do absolute imports.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def pytest_configure(config):
    """注册自定义 mark.

    ``slow``: 标记慢测试 (如圆柱绕流全流程), 默认可用 ``-m 'not slow'`` 跳过.
    """
    config.addinivalue_line(
        "markers",
        "slow: 标记慢测试 (如圆柱绕流全流程, 默认可 -m 'not slow' 跳过)",
    )

