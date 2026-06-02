import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Monolith servers need full runtime deps; modular routes are covered in CI",
)


def _assert_unique_routes(app):
    # Ensure there are no duplicate (path, methods) combinations.
    seen = set()
    duplicates = []
    for route in getattr(app, "routes", []):
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if not methods or not path:
            continue
        for method in sorted(m for m in methods if m not in {"HEAD", "OPTIONS"}):
            key = (method, path)
            if key in seen:
                duplicates.append(key)
            else:
                seen.add(key)
    assert not duplicates, f"Duplicate routes detected: {duplicates}"


def test_dev_app_routes_unique():
    from api.llm_server import app

    _assert_unique_routes(app)


def test_production_app_routes_unique():
    from api.llm_server_production import app

    _assert_unique_routes(app)

