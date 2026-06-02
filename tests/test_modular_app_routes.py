def _route_signatures(app):
    signatures = set()
    for route in getattr(app, "routes", []):
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if not methods or not path:
            continue
        for method in sorted(m for m in methods if m not in {"HEAD", "OPTIONS"}):
            signatures.add((method, path))
    return signatures


def test_modular_app_registers_expected_routes():
    from api.app.main import app

    actual = _route_signatures(app)
    expected = {
        ("GET", "/"),
        ("GET", "/health"),
        ("POST", "/v1/query"),
        ("POST", "/v1/chat/completions"),
        ("DELETE", "/v1/session/{session_id}"),
        ("POST", "/v1/regenerate"),
        ("POST", "/v1/refresh"),
        ("GET", "/v1/debug/search"),
        ("GET", "/v1/debug/find-blog"),
        ("GET", "/v1/search/blogs"),
        ("GET", "/v1/search/schools"),
    }

    missing = expected - actual
    assert not missing, f"Missing modular routes: {sorted(missing)}"


def test_modular_app_routes_are_unique():
    from api.app.main import app

    seen = set()
    duplicates = []
    for signature in sorted(_route_signatures(app)):
        if signature in seen:
            duplicates.append(signature)
        else:
            seen.add(signature)

    assert not duplicates, f"Duplicate modular routes detected: {duplicates}"

