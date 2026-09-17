import sys

import pytest

from podagent import cp
from podagent.ops import pack


@pytest.fixture(autouse=True)
def _fresh_rented_pod_mark():
    # mark_rented_pod() is a one-way module global; a test that boots main() would otherwise
    # poison every later file:// test in the same pytest process (order-dependent CI red).
    cp._RENTED_POD = False
    yield
    cp._RENTED_POD = False


@pytest.fixture(autouse=True)
def _isolate_ops_pack_state():
    # importlib caches packages by NAME: a later test's montyops.* import would resolve against an
    # earlier test's already-cached package object (and its stale __path__) without this reset.
    modules_before = set(sys.modules)
    sys_path_before = list(sys.path)
    yield
    pack.reset_for_tests()
    for name in list(sys.modules):
        if name == "montyops" or name.startswith("montyops."):
            if name not in modules_before:
                del sys.modules[name]
    sys.path[:] = sys_path_before
