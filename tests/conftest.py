import pytest

from podagent import cp


@pytest.fixture(autouse=True)
def _fresh_rented_pod_mark():
    # mark_rented_pod() is a one-way module global; a test that boots main() would otherwise
    # poison every later file:// test in the same pytest process (order-dependent CI red).
    cp._RENTED_POD = False
    yield
    cp._RENTED_POD = False
