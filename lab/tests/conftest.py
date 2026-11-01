"""Shared fixtures: a TPC-H SF 0.01 store and the lab built on it."""
from __future__ import annotations

import pytest

from dal_lab.facades import Catalog, open_store
from dal_lab.policies import Lab
from dal_lab.tpch import ensure_tpch


@pytest.fixture(scope="session")
def con():
    """Read-only session over TPC-H SF 0.01."""
    c = open_store(ensure_tpch(0.01))
    yield c
    c.close()


@pytest.fixture(scope="session")
def catalog(con):
    """Catalog of the SF 0.01 store."""
    return Catalog.load(con)


@pytest.fixture(scope="session")
def lab(con):
    """All facades and scripted policies over SF 0.01."""
    return Lab(con)
